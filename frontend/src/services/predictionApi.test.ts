import { describe, expect, it, vi } from 'vitest'
import {
  buildPredictionUrl,
  PredictionApiError,
  requestPrediction,
} from './predictionApi'
import type { PredictionRequest } from '../types/prediction'

const payload: PredictionRequest = {
  age: 29,
  sex: 'male',
  bmi: 27.9,
  children: 2,
  smoker: 'no',
  region: 'southwest',
}

describe('buildPredictionUrl', () => {
  it('uses a relative path in local development', () => {
    expect(
      buildPredictionUrl({
        isProduction: false,
        configuredBaseUrl: 'https://ignored.example',
      }),
    ).toBe('/predict-json')
  })

  it('uses and normalizes the production base URL', () => {
    expect(
      buildPredictionUrl({
        isProduction: true,
        configuredBaseUrl: 'https://api.example.com///',
      }),
    ).toBe('https://api.example.com/predict-json')
  })

  it.each([
    undefined,
    '',
    'not-a-url',
    'ftp://api.example.com',
    'http://api.example.com',
    'https://api.example.com?token=public',
  ])(
    'returns a controlled configuration error for %s',
    (configuredBaseUrl) => {
      expect(() =>
        buildPredictionUrl({ isProduction: true, configuredBaseUrl }),
      ).toThrow(PredictionApiError)
    },
  )
})

describe('requestPrediction', () => {
  it('posts the exact JSON contract and returns a valid result', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ predicted_charges: 8450.25, currency: 'USD' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    )

    await expect(
      requestPrediction(payload, { isProduction: false }),
    ).resolves.toEqual({ predicted_charges: 8450.25, currency: 'USD' })
    expect(fetchMock).toHaveBeenCalledWith('/predict-json', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  })

  it('maps backend validation details without exposing internal data', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(
        JSON.stringify({
          detail: [{ loc: ['body', 'bmi'], msg: 'private validator detail' }],
        }),
        { status: 422 },
      ),
    )

    const error = await requestPrediction(payload, { isProduction: false }).catch(
      (caught: unknown) => caught,
    )
    expect(error).toBeInstanceOf(PredictionApiError)
    expect(error).toMatchObject({
      kind: 'validation',
      fieldErrors: { bmi: 'BMI was not accepted. Check this value and try again.' },
    })
    expect((error as Error).message).not.toContain('private')
  })

  it.each([
    [500, 'server'],
    [400, 'request'],
  ] as const)('maps HTTP %s to a safe %s error', async (status, kind) => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('{}', { status }))
    await expect(
      requestPrediction(payload, { isProduction: false }),
    ).rejects.toMatchObject({ kind })
  })

  it('handles a network failure', async () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new TypeError('offline'))
    await expect(
      requestPrediction(payload, { isProduction: false }),
    ).rejects.toMatchObject({ kind: 'network' })
  })

  it.each([
    {},
    { predicted_charges: '8450.25', currency: 'USD' },
    { predicted_charges: Number.NaN, currency: 'USD' },
    { predicted_charges: 8450.25, currency: '' },
    { predicted_charges: 8450.25, currency: 'not-a-currency' },
  ])('rejects an invalid successful response %#', async (body) => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify(body), { status: 200 }),
    )
    await expect(
      requestPrediction(payload, { isProduction: false }),
    ).rejects.toMatchObject({ kind: 'invalid-response' })
  })
})
