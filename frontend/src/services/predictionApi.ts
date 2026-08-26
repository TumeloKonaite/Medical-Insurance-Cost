import type {
  FieldErrors,
  PredictionField,
  PredictionRequest,
  PredictionResponse,
} from '../types/prediction'
import { formatCurrency } from '../utils/currency'

export type PredictionErrorKind =
  | 'configuration'
  | 'validation'
  | 'request'
  | 'server'
  | 'network'
  | 'invalid-response'

export class PredictionApiError extends Error {
  constructor(
    public readonly kind: PredictionErrorKind,
    message: string,
    public readonly fieldErrors: FieldErrors = {},
  ) {
    super(message)
    this.name = 'PredictionApiError'
  }
}

interface ApiUrlOptions {
  isProduction?: boolean
  configuredBaseUrl?: string
}

export function buildPredictionUrl({
  isProduction = import.meta.env.PROD,
  configuredBaseUrl = import.meta.env.VITE_API_BASE_URL,
}: ApiUrlOptions = {}): string {
  if (!isProduction) return '/predict-json'

  const candidate = configuredBaseUrl?.trim()
  if (!candidate) {
    throw new PredictionApiError(
      'configuration',
      'The prediction service has not been configured. Please try again later.',
    )
  }

  try {
    const url = new URL(candidate)
    const isSecure =
      url.protocol === 'https:' ||
      (url.protocol === 'http:' && ['localhost', '127.0.0.1'].includes(url.hostname))
    if (
      !isSecure ||
      url.username ||
      url.password ||
      url.search ||
      url.hash
    ) {
      throw new Error('Unsupported API URL')
    }
    return `${url.toString().replace(/\/+$/, '')}/predict-json`
  } catch {
    throw new PredictionApiError(
      'configuration',
      'The prediction service configuration is invalid. Please try again later.',
    )
  }
}

const FIELD_NAMES: Record<PredictionField, string> = {
  age: 'Age',
  sex: 'Sex',
  bmi: 'BMI',
  children: 'Number of children',
  smoker: 'Smoker status',
  region: 'Region',
}

function parseBackendValidation(body: unknown): FieldErrors {
  if (!body || typeof body !== 'object' || !('detail' in body)) return {}
  const detail = (body as { detail?: unknown }).detail
  if (!Array.isArray(detail)) return {}

  const fieldErrors: FieldErrors = {}
  for (const item of detail) {
    if (!item || typeof item !== 'object') continue
    const location = (item as { loc?: unknown }).loc
    if (!Array.isArray(location)) continue
    const field = location.at(-1)
    if (typeof field !== 'string' || !(field in FIELD_NAMES)) continue
    const predictionField = field as PredictionField
    fieldErrors[predictionField] = `${FIELD_NAMES[predictionField]} was not accepted. Check this value and try again.`
  }
  return fieldErrors
}

function isPredictionResponse(value: unknown): value is PredictionResponse {
  if (!value || typeof value !== 'object') return false
  const response = value as Record<string, unknown>
  return (
    typeof response.predicted_charges === 'number' &&
    Number.isFinite(response.predicted_charges) &&
    typeof response.currency === 'string' &&
    formatCurrency(response.predicted_charges, response.currency) !== null
  )
}

async function readJson(response: Response): Promise<unknown> {
  try {
    return await response.json()
  } catch {
    return undefined
  }
}

export async function requestPrediction(
  payload: PredictionRequest,
  options: ApiUrlOptions = {},
): Promise<PredictionResponse> {
  const url = buildPredictionUrl(options)
  let response: Response

  try {
    response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  } catch {
    throw new PredictionApiError(
      'network',
      'We could not reach the prediction service. Check your connection and try again.',
    )
  }

  const body = await readJson(response)

  if (response.status === 422) {
    throw new PredictionApiError(
      'validation',
      'Some details were not accepted. Review the highlighted fields and try again.',
      parseBackendValidation(body),
    )
  }
  if (response.status >= 500) {
    throw new PredictionApiError(
      'server',
      'The prediction service is temporarily unavailable. Please try again shortly.',
    )
  }
  if (!response.ok) {
    throw new PredictionApiError(
      'request',
      'We could not process that request. Review your details and try again.',
    )
  }
  if (!isPredictionResponse(body)) {
    throw new PredictionApiError(
      'invalid-response',
      'We received an unexpected response. Please try again.',
    )
  }

  return body
}
