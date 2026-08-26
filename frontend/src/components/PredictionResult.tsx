import { forwardRef } from 'react'
import type { PredictionResponse } from '../types/prediction'
import { formatCurrency } from '../utils/currency'

interface PredictionResultProps {
  prediction: PredictionResponse
  onReset: () => void
}

export const PredictionResult = forwardRef<HTMLElement, PredictionResultProps>(
  function PredictionResult({ prediction, onReset }, ref) {
    const formatted = formatCurrency(
      prediction.predicted_charges,
      prediction.currency,
    )

    if (!formatted) return null

    return (
      <section
        className="result-card"
        aria-live="polite"
        aria-atomic="true"
        tabIndex={-1}
        ref={ref}
      >
        <div className="result-card__eyebrow">
          <span aria-hidden="true">✓</span>
          Estimate ready
        </div>
        <p className="result-card__label">Estimated annual charge</p>
        <p className="result-card__amount">{formatted}</p>
        <p className="result-card__context">
          That’s approximately{' '}
          {formatCurrency(prediction.predicted_charges / 12, prediction.currency)}{' '}
          per month.
        </p>
        <div className="result-card__divider" />
        <p className="result-card__note">
          This estimate is based on the information you provided and is for
          demonstration purposes only.
        </p>
        <button className="secondary-button" type="button" onClick={onReset}>
          <span aria-hidden="true">↻</span>
          New estimate
        </button>
      </section>
    )
  },
)
