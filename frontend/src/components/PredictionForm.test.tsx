import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { PredictionForm } from './PredictionForm'

async function completeForm() {
  const user = userEvent.setup()
  await user.type(screen.getByLabelText('Age'), '29')
  await user.selectOptions(screen.getByLabelText('Sex'), 'male')
  await user.type(screen.getByLabelText('Body mass index (BMI)'), '27.9')
  await user.type(screen.getByLabelText('Number of children'), '2')
  await user.selectOptions(screen.getByLabelText('Smoker'), 'no')
  await user.selectOptions(screen.getByLabelText('Region'), 'southwest')
  return user
}

describe('PredictionForm', () => {
  it('does not submit an invalid form and shows inline errors', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch')
    const user = userEvent.setup()
    render(<PredictionForm />)

    await user.click(screen.getByRole('button', { name: 'Calculate my estimate' }))

    expect(await screen.findByText('Enter age.')).toBeVisible()
    expect(screen.getByLabelText('Age')).toHaveAttribute('aria-invalid', 'true')
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('shows loading, disables submission, and prevents duplicate requests', async () => {
    let resolveRequest!: (response: Response) => void
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(
      () => new Promise<Response>((resolve) => { resolveRequest = resolve }),
    )
    render(<PredictionForm />)
    const user = await completeForm()
    const submit = screen.getByRole('button', { name: 'Calculate my estimate' })

    await user.dblClick(submit)

    expect(screen.getByRole('button', { name: 'Calculating estimate…' })).toBeDisabled()
    expect(screen.getByRole('status')).toHaveTextContent('being calculated')
    expect(fetchMock).toHaveBeenCalledTimes(1)
    resolveRequest(
      new Response(JSON.stringify({ predicted_charges: 8450.25, currency: 'USD' }), {
        status: 200,
      }),
    )
    expect(await screen.findByText(/8,450\.25/)).toBeVisible()
  })

  it('renders a successful estimate and resets to a focused empty form', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ predicted_charges: 8450.25, currency: 'USD' }), {
        status: 200,
      }),
    )
    render(<PredictionForm />)
    const user = await completeForm()

    await user.click(screen.getByRole('button', { name: 'Calculate my estimate' }))
    expect(await screen.findByText('Estimated annual charge')).toBeVisible()
    expect(screen.getByText(/8,450\.25/)).toBeVisible()

    await user.click(screen.getByRole('button', { name: 'New estimate' }))
    const age = screen.getByLabelText('Age')
    expect(age).toHaveValue('')
    expect(age).toHaveFocus()
    expect(screen.queryByText('Estimated annual charge')).not.toBeInTheDocument()
  })

  it('clears a stale request error when a new request starts', async () => {
    const fetchMock = vi
      .spyOn(globalThis, 'fetch')
      .mockRejectedValueOnce(new TypeError('offline'))
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ predicted_charges: 1000, currency: 'USD' }), {
          status: 200,
        }),
      )
    render(<PredictionForm />)
    const user = await completeForm()
    const submit = screen.getByRole('button', { name: 'Calculate my estimate' })

    await user.click(submit)
    expect(await screen.findByText(/could not reach/)).toBeVisible()
    await user.click(submit)
    await waitFor(() => expect(screen.queryByText(/could not reach/)).toBeNull())
    expect(await screen.findByText(/1,000\.00/)).toBeVisible()
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('surfaces backend field validation safely', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ detail: [{ loc: ['body', 'age'] }] }), {
        status: 422,
      }),
    )
    render(<PredictionForm />)
    const user = await completeForm()

    await user.click(screen.getByRole('button', { name: 'Calculate my estimate' }))

    expect(await screen.findByText(/Some details were not accepted/)).toBeVisible()
    expect(screen.getByText(/Age was not accepted/)).toBeVisible()
  })
})
