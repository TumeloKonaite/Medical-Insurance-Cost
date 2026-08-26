import type { ReactNode } from 'react'

interface FormFieldProps {
  id: string
  label: string
  hint?: string
  error?: string
  children: ReactNode
  className?: string
}

export function FormField({
  id,
  label,
  hint,
  error,
  children,
  className = '',
}: FormFieldProps) {
  return (
    <div className={`form-field ${className}`.trim()}>
      <label htmlFor={id}>{label}</label>
      {hint && (
        <span className="field-hint" id={`${id}-hint`}>
          {hint}
        </span>
      )}
      {children}
      {error && (
        <span className="field-error" id={`${id}-error`} role="alert">
          <span aria-hidden="true">!</span>
          {error}
        </span>
      )}
    </div>
  )
}
