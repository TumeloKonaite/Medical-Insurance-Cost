import { useEffect, useRef, useState, type ChangeEvent, type FormEvent } from 'react'
import { requestPrediction, PredictionApiError } from '../services/predictionApi'
import type {
  FieldErrors,
  PredictionField,
  PredictionFormValues,
  PredictionResponse,
} from '../types/prediction'
import { validatePredictionForm } from '../utils/validation'
import { ErrorMessage } from './ErrorMessage'
import { FormField } from './FormField'
import { PredictionResult } from './PredictionResult'

const INITIAL_VALUES: PredictionFormValues = {
  age: '',
  sex: '',
  bmi: '',
  children: '',
  smoker: '',
  region: '',
}

function describedBy(id: string, hasError: boolean, hasHint = false) {
  return [hasHint ? `${id}-hint` : '', hasError ? `${id}-error` : '']
    .filter(Boolean)
    .join(' ') || undefined
}

export function PredictionForm() {
  const [values, setValues] = useState<PredictionFormValues>(INITIAL_VALUES)
  const [fieldErrors, setFieldErrors] = useState<FieldErrors>({})
  const [requestError, setRequestError] = useState('')
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [resetCount, setResetCount] = useState(0)
  const submissionLock = useRef(false)
  const firstFieldRef = useRef<HTMLInputElement>(null)
  const resultRef = useRef<HTMLElement>(null)
  const errorRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (prediction) resultRef.current?.focus()
  }, [prediction])

  useEffect(() => {
    if (requestError) errorRef.current?.focus()
  }, [requestError])

  useEffect(() => {
    if (resetCount > 0) firstFieldRef.current?.focus()
  }, [resetCount])

  const updateField = (
    event: ChangeEvent<HTMLInputElement | HTMLSelectElement>,
  ) => {
    const field = event.target.name as PredictionField
    setValues((current) => ({ ...current, [field]: event.target.value }))
    setFieldErrors((current) => {
      if (!current[field]) return current
      const next = { ...current }
      delete next[field]
      return next
    })
  }

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (submissionLock.current) return

    const validation = validatePredictionForm(values)
    setFieldErrors(validation.errors)
    if (!validation.payload) {
      const firstInvalid = Object.keys(validation.errors)[0] as PredictionField
      document.getElementById(firstInvalid)?.focus()
      return
    }

    submissionLock.current = true
    setIsLoading(true)
    setRequestError('')
    setPrediction(null)

    try {
      const response = await requestPrediction(validation.payload)
      setPrediction(response)
    } catch (error) {
      if (error instanceof PredictionApiError) {
        setRequestError(error.message)
        if (error.kind === 'validation') {
          setFieldErrors(error.fieldErrors)
        }
      } else {
        setRequestError('Something unexpected happened. Please try again.')
      }
    } finally {
      submissionLock.current = false
      setIsLoading(false)
    }
  }

  const handleReset = () => {
    submissionLock.current = false
    setValues(INITIAL_VALUES)
    setFieldErrors({})
    setRequestError('')
    setPrediction(null)
    setIsLoading(false)
    setResetCount((count) => count + 1)
  }

  if (prediction) {
    return (
      <PredictionResult
        prediction={prediction}
        onReset={handleReset}
        ref={resultRef}
      />
    )
  }

  return (
    <>
      <form noValidate onSubmit={handleSubmit} aria-busy={isLoading}>
        <div className="form-heading">
          <div>
            <span className="step-label">Your details</span>
            <h2>Calculate your estimate</h2>
          </div>
          <span className="time-note" aria-label="Takes about one minute">
            <span aria-hidden="true">◷</span> 1 min
          </span>
        </div>

        {requestError && <ErrorMessage message={requestError} ref={errorRef} />}

        <div className="form-grid">
          <FormField id="age" label="Age" error={fieldErrors.age}>
            <input
              ref={firstFieldRef}
              id="age"
              name="age"
              type="text"
              inputMode="numeric"
              autoComplete="off"
              placeholder="e.g. 34"
              value={values.age}
              onChange={updateField}
              aria-invalid={Boolean(fieldErrors.age)}
              aria-describedby={describedBy('age', Boolean(fieldErrors.age))}
            />
          </FormField>

          <FormField id="sex" label="Sex" error={fieldErrors.sex}>
            <select
              id="sex"
              name="sex"
              autoComplete="sex"
              value={values.sex}
              onChange={updateField}
              aria-invalid={Boolean(fieldErrors.sex)}
              aria-describedby={describedBy('sex', Boolean(fieldErrors.sex))}
            >
              <option value="">Select an option</option>
              <option value="female">Female</option>
              <option value="male">Male</option>
            </select>
          </FormField>

          <FormField
            id="bmi"
            label="Body mass index (BMI)"
            hint="A number greater than 0"
            error={fieldErrors.bmi}
          >
            <input
              id="bmi"
              name="bmi"
              type="text"
              inputMode="decimal"
              autoComplete="off"
              placeholder="e.g. 24.7"
              value={values.bmi}
              onChange={updateField}
              aria-invalid={Boolean(fieldErrors.bmi)}
              aria-describedby={describedBy('bmi', Boolean(fieldErrors.bmi), true)}
            />
          </FormField>

          <FormField
            id="children"
            label="Number of children"
            error={fieldErrors.children}
          >
            <input
              id="children"
              name="children"
              type="text"
              inputMode="numeric"
              autoComplete="off"
              placeholder="e.g. 2"
              value={values.children}
              onChange={updateField}
              aria-invalid={Boolean(fieldErrors.children)}
              aria-describedby={describedBy(
                'children',
                Boolean(fieldErrors.children),
              )}
            />
          </FormField>

          <FormField id="smoker" label="Smoker" error={fieldErrors.smoker}>
            <select
              id="smoker"
              name="smoker"
              autoComplete="off"
              value={values.smoker}
              onChange={updateField}
              aria-invalid={Boolean(fieldErrors.smoker)}
              aria-describedby={describedBy('smoker', Boolean(fieldErrors.smoker))}
            >
              <option value="">Select an option</option>
              <option value="no">No</option>
              <option value="yes">Yes</option>
            </select>
          </FormField>

          <FormField id="region" label="Region" error={fieldErrors.region}>
            <select
              id="region"
              name="region"
              autoComplete="address-level1"
              value={values.region}
              onChange={updateField}
              aria-invalid={Boolean(fieldErrors.region)}
              aria-describedby={describedBy('region', Boolean(fieldErrors.region))}
            >
              <option value="">Select a region</option>
              <option value="northeast">Northeast</option>
              <option value="northwest">Northwest</option>
              <option value="southeast">Southeast</option>
              <option value="southwest">Southwest</option>
            </select>
          </FormField>
        </div>

        <button className="primary-button" type="submit" disabled={isLoading}>
          {isLoading ? (
            <>
              <span className="spinner" aria-hidden="true" />
              Calculating estimate…
            </>
          ) : (
            <>
              Calculate my estimate
              <span aria-hidden="true">→</span>
            </>
          )}
        </button>
        <p className="privacy-note">
          <span aria-hidden="true">◇</span>
          No identifying information is collected.
        </p>
        {isLoading && (
          <span className="sr-only" role="status" aria-live="polite">
            Your estimate is being calculated.
          </span>
        )}
      </form>
    </>
  )
}
