import { Disclaimer } from './components/Disclaimer'
import { PredictionForm } from './components/PredictionForm'

function CheckIcon() {
  return (
    <span className="check-icon" aria-hidden="true">
      ✓
    </span>
  )
}

export default function App() {
  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="#main" aria-label="MedEstimate home">
          <span className="brand-mark" aria-hidden="true">
            <span />
            <span />
          </span>
          <span>MedEstimate</span>
        </a>
        <div className="demo-badge">
          <span aria-hidden="true" /> Demo tool
        </div>
      </header>

      <main id="main" className="main-layout">
        <section className="intro" aria-labelledby="page-title">
          <p className="eyebrow">Insurance cost estimator</p>
          <h1 id="page-title">
            A clearer look at your{' '}
            <span>potential health costs.</span>
          </h1>
          <p className="intro-copy">
            Enter a few details to receive an instant, model-based estimate of
            annual medical insurance charges.
          </p>

          <ul className="benefit-list">
            <li>
              <CheckIcon />
              <span>
                <strong>Quick and simple</strong>
                Get an estimate in under a minute.
              </span>
            </li>
            <li>
              <CheckIcon />
              <span>
                <strong>Private by design</strong>
                No name, email, or account required.
              </span>
            </li>
            <li>
              <CheckIcon />
              <span>
                <strong>Easy to understand</strong>
                A straightforward annual cost estimate.
              </span>
            </li>
          </ul>

          <Disclaimer />
        </section>

        <section className="estimator-panel" aria-label="Insurance cost estimator">
          <PredictionForm />
        </section>
      </main>

      <footer>
        <p>Built as a machine-learning demonstration.</p>
        <p>© {new Date().getFullYear()} MedEstimate</p>
      </footer>
    </div>
  )
}
