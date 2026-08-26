import { describe, expect, it } from 'vitest'
import { formatCurrency } from './currency'

describe('formatCurrency', () => {
  it('formats the API amount using its returned currency', () => {
    expect(formatCurrency(8450.25, 'USD')).toMatch(/8,450\.25/)
  })

  it('fails safely for invalid values and currency codes', () => {
    expect(formatCurrency(Number.NaN, 'USD')).toBeNull()
    expect(formatCurrency(42, 'not-a-currency')).toBeNull()
  })
})
