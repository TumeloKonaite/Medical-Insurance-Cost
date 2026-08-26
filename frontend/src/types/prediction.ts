export const SEX_VALUES = ['female', 'male'] as const
export const SMOKER_VALUES = ['yes', 'no'] as const
export const REGION_VALUES = [
  'northeast',
  'northwest',
  'southeast',
  'southwest',
] as const

export type Sex = (typeof SEX_VALUES)[number]
export type Smoker = (typeof SMOKER_VALUES)[number]
export type Region = (typeof REGION_VALUES)[number]

export interface PredictionRequest {
  age: number
  sex: Sex
  bmi: number
  children: number
  smoker: Smoker
  region: Region
}

export interface PredictionResponse {
  predicted_charges: number
  currency: string
}

export interface PredictionFormValues {
  age: string
  sex: string
  bmi: string
  children: string
  smoker: string
  region: string
}

export type PredictionField = keyof PredictionFormValues
export type FieldErrors = Partial<Record<PredictionField, string>>
