import { forwardRef } from 'react'

interface ErrorMessageProps {
  message: string
}

export const ErrorMessage = forwardRef<HTMLDivElement, ErrorMessageProps>(
  function ErrorMessage({ message }, ref) {
    return (
      <div
        className="request-error"
        role="alert"
        aria-live="assertive"
        tabIndex={-1}
        ref={ref}
      >
        <span className="request-error__icon" aria-hidden="true">
          !
        </span>
        <div>
          <strong>We couldn’t complete your estimate</strong>
          <p>{message}</p>
        </div>
      </div>
    )
  },
)
