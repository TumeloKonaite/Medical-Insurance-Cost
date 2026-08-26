import { describe, expect, it } from 'vitest'
import { validatePredictionForm } from './validation'
import type { PredictionFormValues } from '../types/prediction'

const validValues: PredictionFormValues = {
  age: '29',
  sex: 'male',
  bmi: '27.9',
  children: '2',
  smoker: 'no',
  region: 'southwest',
}

describe('validatePredictionForm', () => {
  it('requires every field without converting empty numbers to zero', () => {
    const result = validatePredictionForm({
      age: '',
      sex: '',
      bmi: '',
      children: '',
      smoker: '',
      region: '',
    })

    expect(result.payload).toBeUndefined()
    expect(Object.keys(result.errors)).toHaveLength(6)
    expect(result.errors.age).toBe('Enter age.')
    expect(result.errors.children).toBe('Enter number of children.')
  })

  it.each([
    ['age', '-1', 'Age cannot be negative.'],
    ['age', '21.5', 'Age must be a whole number.'],
    ['age', 'Infinity', 'Age must be a finite number.'],
    ['bmi', '0', 'BMI must be greater than 0.'],
    ['bmi', 'NaN', 'BMI must be a finite number.'],
    ['children', '1.5', 'Number of children must be a whole number.'],
  ] as const)('rejects invalid numeric %s values', (field, value, message) => {
    const result = validatePredictionForm({ ...validValues, [field]: value })
    expect(result.errors[field]).toBe(message)
    expect(result.payload).toBeUndefined()
  })

  it.each([
    ['sex', 'other'],
    ['smoker', 'sometimes'],
    ['region', 'central'],
  ] as const)('rejects unsupported %s values', (field, value) => {
    const result = validatePredictionForm({ ...validValues, [field]: value })
    expect(result.errors[field]).toBeDefined()
    expect(result.payload).toBeUndefined()
  })

  it('converts valid numeric strings into the typed API payload', () => {
    expect(validatePredictionForm(validValues).payload).toEqual({
      age: 29,
      sex: 'male',
      bmi: 27.9,
      children: 2,
      smoker: 'no',
      region: 'southwest',
    })
  })
})
