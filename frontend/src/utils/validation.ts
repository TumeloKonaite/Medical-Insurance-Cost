import {
  REGION_VALUES,
  SEX_VALUES,
  SMOKER_VALUES,
  type FieldErrors,
  type PredictionFormValues,
  type PredictionRequest,
  type Region,
  type Sex,
  type Smoker,
} from '../types/prediction'

export interface ValidationResult {
  errors: FieldErrors
  payload?: PredictionRequest
}

function parseInteger(
  value: string,
  label: string,
): { value?: number; error?: string } {
  if (value.trim() === '') {
    return { error: `Enter ${label.toLowerCase()}.` }
  }

  const parsed = Number(value)
  if (!Number.isFinite(parsed)) {
    return { error: `${label} must be a finite number.` }
  }
  if (!Number.isInteger(parsed)) {
    return { error: `${label} must be a whole number.` }
  }
  if (parsed < 0) {
    return { error: `${label} cannot be negative.` }
  }
  return { value: parsed }
}

function parseBmi(value: string): { value?: number; error?: string } {
  if (value.trim() === '') {
    return { error: 'Enter your BMI.' }
  }

  const parsed = Number(value)
  if (!Number.isFinite(parsed)) {
    return { error: 'BMI must be a finite number.' }
  }
  if (parsed <= 0) {
    return { error: 'BMI must be greater than 0.' }
  }
  return { value: parsed }
}

function isIncluded<T extends string>(
  values: readonly T[],
  value: string,
): value is T {
  return values.includes(value as T)
}

export function validatePredictionForm(
  values: PredictionFormValues,
): ValidationResult {
  const errors: FieldErrors = {}
  const age = parseInteger(values.age, 'Age')
  const bmi = parseBmi(values.bmi)
  const children = parseInteger(values.children, 'Number of children')

  if (age.error) errors.age = age.error
  if (bmi.error) errors.bmi = bmi.error
  if (children.error) errors.children = children.error

  if (!isIncluded(SEX_VALUES, values.sex)) {
    errors.sex = 'Choose female or male.'
  }
  if (!isIncluded(SMOKER_VALUES, values.smoker)) {
    errors.smoker = 'Choose yes or no.'
  }
  if (!isIncluded(REGION_VALUES, values.region)) {
    errors.region = 'Choose a region from the list.'
  }

  if (Object.keys(errors).length > 0) return { errors }

  return {
    errors,
    payload: {
      age: age.value as number,
      sex: values.sex as Sex,
      bmi: bmi.value as number,
      children: children.value as number,
      smoker: values.smoker as Smoker,
      region: values.region as Region,
    },
  }
}
