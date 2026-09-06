import { useEffect, useMemo, useState } from 'react';
import { Navigate, NavLink, Route, Routes, useLocation, useNavigate } from 'react-router-dom';
import {
  Activity,
  ArrowRight,
  BarChart3,
  CheckCircle2,
  ChevronDown,
  CircleHelp,
  ClipboardList,
  Code2,
  FileJson,
  Gauge,
  Home as HomeIcon,
  Info,
  LockKeyhole,
  Menu,
  Minus,
  Plus,
  RotateCcw,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Target,
  TrendingUp,
  UserRound,
  WalletCards,
  XCircle,
  Zap,
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';
const STORAGE_KEY = 'loanlens:last-analysis';

const DEFAULT_FORM = {
  Gender: 'Male',
  Married: 'Married',
  Dependents: '0',
  Education: 'Graduate',
  Employment_Status: 'Salaried',
  Applicant_Income: 50000,
  Coapplicant_Income: 20000,
  Loan_Amount: 150000,
  Loan_Term: 360,
  Credit_History: 1,
  Property_Area: 'Urban',
  Age: 32,
};

function saveAnalysis(analysis, application) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify({ analysis, application }));
}

function loadAnalysis() {
  try {
    const value = localStorage.getItem(STORAGE_KEY);
    return value ? JSON.parse(value) : null;
  } catch {
    return null;
  }
}

function formatMoney(value) {
  return `₹${Number(value || 0).toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
}

function formatValue(value) {
  if (typeof value === 'number') return Number.isInteger(value) ? value.toLocaleString('en-IN') : value.toFixed(2);
  return String(value);
}

function featureLabel(feature) {
  return feature.replaceAll('_', ' ');
}

function Logo() {
  return (
    <div className="brand">
      <div className="brand-mark"><Activity size={17} /></div>
      <div>
        <div className="brand-name">LoanLens <span className="api-dot">● API CONNECTED</span></div>
        <div className="brand-sub">Explainable credit decisions</div>
      </div>
    </div>
  );
}

function Header({ section }) {
  return (
    <header className="topbar">
      <Logo />
      <div className="header-section">{section}</div>
      <div className="profile"><UserRound size={17} /></div>
    </header>
  );
}

function BottomNav() {
  const items = [
    ['home', 'Home', HomeIcon],
    ['apply', 'Apply', ClipboardList],
    ['decision', 'Decision', BarChart3],
    ['what-if', 'What-If', SlidersHorizontal],
  ];

  return (
    <nav className="bottom-nav">
      {items.map(([to, label, Icon]) => (
        <NavLink key={to} to={`/${to}`} className={({ isActive }) => isActive ? 'nav-item active' : 'nav-item'}>
          <Icon size={20} />
          <span>{label}</span>
        </NavLink>
      ))}
    </nav>
  );
}

function Layout({ children, section }) {
  return (
    <div className="app-shell">
      <Header section={section} />
      <main className="page">{children}</main>
      <BottomNav />
    </div>
  );
}

function Landing() {
  const navigate = useNavigate();
  return (
    <div className="landing">
      <div className="landing-grid" />
      <header className="landing-header"><Logo /><span className="landing-badge"><Sparkles size={13} /> Explainable AI</span></header>
      <main className="landing-content">
        <div className="eyebrow"><span className="pulse-dot" /> AI FOR ENGINEERS • LOANLENS</div>
        <h1>See <span>why</span> a loan decision happens.</h1>
        <p className="landing-lead">
          LoanLens turns a loan application into an interpretable model decision — with feature-level explanations and realistic what-if scenarios.
        </p>
        <div className="landing-actions">
          <button className="btn primary big" onClick={() => navigate('/apply')}>Analyze a Loan <ArrowRight size={19} /></button>
          <button className="btn ghost big" onClick={() => navigate('/home')}>Explore LoanLens</button>
        </div>
        <div className="landing-stats">
          <div><strong>12</strong><span>input parameters</span></div>
          <div><strong>SHAP</strong><span>decision explanations</span></div>
          <div><strong>What-If</strong><span>actionable scenarios</span></div>
        </div>
        <div className="landing-note"><ShieldCheck size={17} /> Built as an explainable AI simulator — not a real bank underwriting system.</div>
      </main>
      <footer className="landing-footer">LoanLens • Explainable AI Credit Decision Simulator</footer>
    </div>
  );
}

function Home() {
  const navigate = useNavigate();
  const saved = loadAnalysis();
  const lastProbability = saved?.analysis?.approval_probability;
  return (
    <Layout section="Home">
      <section className="hero-card">
        <div className="eyebrow"><span className="pulse-dot" /> ENGINE PIPELINE ACTIVE • EXPLAINABLE AI</div>
        <h1>Predict. Explain. <span>Explore Alternatives.</span></h1>
        <p>LoanLens is a transparent credit decision simulator. Connect an application to the model for probabilistic decisioning, feature contributions, and actionable counterfactuals.</p>
        <div className="feature-chips">
          <span><Zap size={14} /> Instant decision</span>
          <span><BarChart3 size={14} /> SHAP contributions</span>
          <span><SlidersHorizontal size={14} /> What-If scenarios</span>
          <span><LockKeyhole size={14} /> Local API</span>
        </div>
        <button className="btn primary full" onClick={() => navigate('/apply')}>Analyze Loan Application <ArrowRight size={18} /></button>
        <button className="btn secondary full" onClick={() => saved ? navigate('/decision') : navigate('/apply')}>View Analysis Results</button>
        <div className="telemetry">
          <div className="telemetry-head"><span><Activity size={14} /> LIVE PIPELINE</span><span>POST /analyze</span></div>
          <div className="telemetry-bars"><i /><i /><i /></div>
          <div className="telemetry-labels"><span>Predict</span><span>Explain</span><span>What-If</span></div>
          {lastProbability != null && <div className="last-run">Last result: <b>{(lastProbability * 100).toFixed(2)}% approval probability</b></div>}
        </div>
      </section>

      <section className="section-block">
        <div className="eyebrow">INSTITUTIONAL RELIABILITY</div>
        <h2>Why choose LoanLens?</h2>
        <div className="feature-grid">
          <InfoCard icon={Zap} title="Instant Decision" text="Fast model evaluation across all 12 verified application parameters." />
          <InfoCard icon={BarChart3} title="Feature Contributions" text="See which input factors support or oppose the model's decision." />
          <InfoCard icon={SlidersHorizontal} title="What-If Delta" text="Test realistic actionable changes against the model decision boundary." />
          <InfoCard icon={ShieldCheck} title="Zero Black Box" text="The prediction is paired with an interpretable explanation instead of a bare score." />
        </div>
      </section>

      <section className="section-block">
        <div className="eyebrow">ALGORITHMIC CLARITY</div>
        <h2>Frequently Asked Questions</h2>
        <Faq question="How does the LoanLens decision engine work?" answer="Your 12 application fields are sent to the trained Logistic Regression pipeline. The API returns the predicted class, approval probability, SHAP contributions, and constrained counterfactuals." />
        <Faq question="What are the 12 required parameters?" answer="Gender, marital status, dependents, education, employment status, applicant income, coapplicant income, loan amount, loan term, credit history, property area, and age." />
        <Faq question="What is an actionable counterfactual scenario?" answer="A hypothetical change to an actionable input that flips the model prediction within the realistic ranges tested by LoanLens. Credit History is deliberately held fixed." />
        <Faq question="Is this an actual HDFC loan approval system?" answer="No. LoanLens uses a synthetic Kaggle dataset that simulates HDFC loan decisions. Results are for educational and demonstration purposes." />
      </section>

      <button className="ready-card" onClick={() => navigate('/apply')}>
        <div className="ready-icon"><Target size={20} /></div>
        <div><b>Ready to evaluate application parameters?</b><span>Proceed to the 12-input analysis form</span></div>
        <ArrowRight size={20} />
      </button>
    </Layout>
  );
}

function InfoCard({ icon: Icon, title, text }) {
  return <div className="info-card"><div className="info-icon"><Icon size={18} /></div><b>{title}</b><p>{text}</p></div>;
}

function Faq({ question, answer }) {
  const [open, setOpen] = useState(false);
  return <div className="faq"><button onClick={() => setOpen(!open)}><span>{question}</span><ChevronDown className={open ? 'rotate' : ''} size={18} /></button>{open && <p>{answer}</p>}</div>;
}

function PillGroup({ label, value, options, onChange }) {
  return <div className="field"><label>{label}</label><div className="pills">{options.map(option => <button type="button" key={option} className={value === option ? 'pill selected' : 'pill'} onClick={() => onChange(option)}>{option}</button>)}</div></div>;
}

function NumberField({ label, value, onChange, suffix, min = 0, step = 1, helper }) {
  return <div className="field"><label>{label}{helper && <small>{helper}</small>}</label><div className="input-wrap"><input type="number" value={value} min={min} step={step} onChange={e => onChange(Number(e.target.value))} />{suffix && <span>{suffix}</span>}</div></div>;
}

function Apply() {
  const navigate = useNavigate();
  const [form, setForm] = useState(DEFAULT_FORM);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const set = (key, value) => setForm(prev => ({ ...prev, [key]: value }));

  async function analyze() {
    setLoading(true);
    setError('');
    try {
      const response = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail?.[0]?.msg || data.detail || 'The API rejected this application.');
      saveAnalysis(data, form);
      navigate('/decision');
    } catch (err) {
      setError(err.message || 'Unable to reach LoanLens API.');
    } finally {
      setLoading(false);
    }
  }

  function demoFill() {
    setForm({ ...DEFAULT_FORM, Applicant_Income: 85000, Coapplicant_Income: 35000, Loan_Amount: 320000, Age: 29 });
  }

  return (
    <Layout section="Apply">
      <div className="progress-card"><div><span className="eyebrow"><span className="pulse-dot" /> STEP 1 • DIGITAL PROFILE INTAKE</span><button className="demo-btn" onClick={demoFill}><Zap size={13} /> Demo Fill</button></div><div className="progress-line"><span>Feature Matrix Normalization</span><b>12 / 12 Ready</b></div><div className="progress"><i /></div></div>
      <div className="title-row"><div><div className="eyebrow"><Activity size={15} /> POST /analyze READY</div><h1>Algorithmic Underwrite</h1><p>Enter the same 12 parameters expected by the LoanLens model.</p></div></div>

      <section className="form-card">
        <div className="card-title"><ShieldCheck size={19} /><h2>Demographic Profile</h2></div>
        <PillGroup label="1. Applicant Gender" value={form.Gender} options={['Male', 'Female', 'Other']} onChange={v => set('Gender', v)} />
        <PillGroup label="2. Marital Status" value={form.Married} options={['Married', 'Single']} onChange={v => set('Married', v)} />
        <PillGroup label="3. Number of Dependents" value={form.Dependents} options={['0', '1', '2', '3+']} onChange={v => set('Dependents', v)} />
        <div className="field"><label>4. Applicant Age</label><div className="stepper"><button onClick={() => set('Age', Math.max(1, form.Age - 1))}><Minus size={18} /></button><strong>{form.Age}</strong><span>Years</span><button onClick={() => set('Age', Math.min(100, form.Age + 1))}><Plus size={18} /></button></div></div>
      </section>

      <section className="form-card">
        <div className="card-title"><WalletCards size={19} /><h2>Income & Employment</h2></div>
        <PillGroup label="5. Education" value={form.Education} options={['Graduate', 'Not Graduate']} onChange={v => set('Education', v)} />
        <PillGroup label="6. Employment Status" value={form.Employment_Status} options={['Salaried', 'Self-Employed', 'Unemployed']} onChange={v => set('Employment_Status', v)} />
        <NumberField label="7. Applicant Income" value={form.Applicant_Income} onChange={v => set('Applicant_Income', v)} suffix="₹ / mo" />
        <NumberField label="8. Coapplicant Income" value={form.Coapplicant_Income} onChange={v => set('Coapplicant_Income', v)} suffix="₹ / mo" helper="Enter 0 if there is no coapplicant" />
      </section>

      <section className="form-card">
        <div className="card-title"><Gauge size={19} /><h2>Credit & Loan Setup</h2></div>
        <NumberField label="9. Requested Loan Amount" value={form.Loan_Amount} onChange={v => set('Loan_Amount', v)} suffix="₹" />
        <PillGroup label="10. Tenure Duration" value={String(form.Loan_Term)} options={['60', '120', '180', '240', '360']} onChange={v => set('Loan_Term', Number(v))} />
        <PillGroup label="11. Credit History" value={String(form.Credit_History)} options={['1', '0']} onChange={v => set('Credit_History', Number(v))} />
        <p className="field-note"><Info size={15} /> 1 = positive credit history; 0 = no positive credit history. This feature is kept fixed during counterfactual search.</p>
        <PillGroup label="12. Property Area" value={form.Property_Area} options={['Urban', 'Semiurban', 'Rural']} onChange={v => set('Property_Area', v)} />
      </section>

      <div className="secure-note"><LockKeyhole size={18} /><span>Data is sent only to your local LoanLens FastAPI analysis endpoint.</span></div>
      {error && <div className="error-box"><XCircle size={18} /><span>{error}</span></div>}
      <button className="btn primary analyze-btn" onClick={analyze} disabled={loading}>{loading ? <><RotateCcw className="spin" size={19} /> Evaluating Model...</> : <>Analyze Application <ArrowRight size={19} /></>}</button>
      <p className="form-disclaimer">LoanLens is an educational simulator using a synthetic dataset. A model prediction is not a real bank approval or financial advice.</p>
    </Layout>
  );
}

function Decision() {
  const saved = loadAnalysis();
  const navigate = useNavigate();
  if (!saved) return <EmptyState title="No analysis yet" text="Submit an application first so LoanLens can generate a decision." action="Analyze Application" onClick={() => navigate('/apply')} />;
  const { analysis, application } = saved;
  const approved = analysis.prediction === 'Approved';
  const probability = analysis.approval_probability * 100;
  const explanations = [...analysis.explanations].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  return (
    <Layout section="Decision">
      <div className="status-strip"><span className={approved ? 'status-dot good' : 'status-dot bad'} /> MODEL RESULT • POST /analyze <span className="mono">Decision strength: {analysis.decision_strength}</span></div>
      <section className={`decision-card ${approved ? 'approved' : 'rejected'}`}>
        <div className="decision-badges"><span><Activity size={14} /> LOANLENS DECISION ENGINE</span><b>{approved ? 'APPROVAL' : 'REJECTION'}</b></div>
        <div className="decision-heading"><div className="decision-icon">{approved ? <CheckCircle2 size={30} /> : <XCircle size={30} />}</div><div><h1>LOAN {approved ? 'APPROVED' : 'REJECTED'}</h1><p>Classified by the production Logistic Regression pipeline.</p></div></div>
        <div className="probability-head"><span>Approval Probability</span><strong>{probability.toFixed(2)}%</strong><small>threshold 50.00%</small></div>
        <div className="probability-track"><i style={{ width: `${probability}%` }} /><span /></div>
        <div className="probability-foot"><span>0% Reject</span><b>Margin: {analysis.decision_margin >= 0 ? '+' : ''}{(analysis.decision_margin * 100).toFixed(2)} pp</b><span>100% Approval</span></div>
        <div className="decision-metrics"><Metric label="Decision Strength" value={analysis.decision_strength} /><Metric label="Prediction" value={analysis.prediction} /></div>
      </section>

      <section className="section-block"><div className="section-heading"><div><h2>Why did the model decide this?</h2><p>SHAP contributions show how each feature moved the model's decision score.</p></div><Info size={17} /></div><div className="legend"><span className="positive" /> Supports decision (+) <span className="negative" /> Opposes decision (−)</div>
        <div className="explanation-list">{explanations.slice(0, 8).map((item, index) => <ExplanationCard key={`${item.feature}-${index}`} item={item} />)}</div>
      </section>

      <section className="section-block"><h2>Application Attributes</h2><div className="attribute-grid">
        <Attribute label="Applicant Income" value={formatMoney(application.Applicant_Income)} />
        <Attribute label="Coapplicant Income" value={formatMoney(application.Coapplicant_Income)} />
        <Attribute label="Loan Amount" value={formatMoney(application.Loan_Amount)} />
        <Attribute label="Loan Term" value={`${application.Loan_Term} Months`} />
        <Attribute label="Credit History" value={application.Credit_History === 1 ? 'Positive (1)' : 'Not Positive (0)'} />
        <Attribute label="Employment" value={application.Employment_Status} />
        <Attribute label="Property Area" value={application.Property_Area} />
        <Attribute label="Age" value={`${application.Age} Years`} />
      </div></section>

      <button className="btn secondary full" onClick={() => navigate('/what-if')}>Explore What-If Scenarios <SlidersHorizontal size={18} /></button>
      <button className="btn ghost full" onClick={() => downloadJSON({ analysis, application })}><FileJson size={18} /> Export Analysis JSON</button>
    </Layout>
  );
}

function Metric({ label, value }) { return <div><span>{label}</span><b>{value}</b></div>; }
function Attribute({ label, value }) { return <div className="attribute"><span>{label}</span><b>{value}</b></div>; }

function ExplanationCard({ item }) {
  const positive = item.contribution >= 0;
  const width = Math.min(100, Math.max(8, Math.abs(item.contribution) / 2 * 100));
  return <div className="explanation"><div className="explanation-top"><span>{featureLabel(item.feature)}</span><b className={positive ? 'positive-text' : 'negative-text'}>{positive ? '+' : '−'}{Math.abs(item.contribution).toFixed(3)}</b></div><div className="impact-track"><i className={positive ? 'positive-fill' : 'negative-fill'} style={{ width: `${width}%` }} /></div><div className="explanation-bottom"><span>{formatValue(item.value)}</span><p>{item.impact}</p></div></div>;
}

function WhatIf() {
  const navigate = useNavigate();
  const saved = loadAnalysis();
  if (!saved) return <EmptyState title="No analysis yet" text="Run an application analysis before exploring counterfactuals." action="Analyze Application" onClick={() => navigate('/apply')} />;
  const { analysis } = saved;
  const counterfactuals = analysis.counterfactuals || [];
  const current = analysis.original_probability;
  return (
    <Layout section="What If">
      <div className="status-strip"><span className="status-dot good" /> COUNTERFACTUAL ENGINE ACTIVE <span className="mono">Actionable features only</span></div>
      <section className="whatif-intro"><div className="eyebrow"><SlidersHorizontal size={15} /> WHAT-IF ANALYSIS</div><h1>Could the decision change?</h1><p>LoanLens searches constrained, realistic changes to actionable inputs. Credit History remains fixed and no scenario is presented as a guaranteed bank requirement.</p></section>
      <div className="compare-grid"><div><span>Current State</span><strong className={current >= .5 ? 'positive-text' : 'negative-text'}>{(current * 100).toFixed(2)}%</strong><small>{analysis.original_prediction}</small></div><div><span>Scenarios Found</span><strong>{counterfactuals.length}</strong><small>{analysis.counterfactual_found ? 'Actionable alternatives available' : 'No realistic flip found'}</small></div></div>

      {counterfactuals.length > 0 ? <section className="section-block"><div className="section-heading"><div><h2>Counterfactual Pathway</h2><p>Each card is a model simulation, not a loan approval promise.</p></div><Target size={19} /></div><div className="counterfactual-list">{counterfactuals.slice(0, 5).map((cf, i) => <CounterfactualCard key={i} cf={cf} />)}</div></section> : <div className="fallback"><ShieldCheck size={25} /><h2>No realistic actionable scenario found</h2><p>{analysis.counterfactual_message || 'The model did not find a tested actionable change that flips the prediction within the configured ranges.'}</p>{analysis.best_tested_probability != null && <b>Best tested approval probability: {(analysis.best_tested_probability * 100).toFixed(2)}%</b>}</div>}

      <div className="whatif-note"><Info size={16} /><span>Counterfactuals are hypothetical model behavior. They are not guaranteed approval requirements, financial advice, or causal claims.</span></div>
      <button className="btn secondary full" onClick={() => navigate('/apply')}>Re-evaluate with New Parameters <RotateCcw size={17} /></button>
      <button className="btn ghost full" onClick={() => downloadJSON(saved)}><Code2 size={17} /> Export Analysis JSON</button>
    </Layout>
  );
}

function CounterfactualCard({ cf }) {
  const gain = cf.probability_gain * 100;
  return <div className="cf-card"><div className="cf-head"><span>Scenario {cf.counterfactual_type || 'Actionable'}</span><b>+{gain.toFixed(2)} pp</b></div><div className="cf-result"><strong>{(cf.new_probability * 100).toFixed(2)}%</strong><span>{cf.new_prediction}</span></div><div className="cf-changes">{cf.changes.map((change, index) => <div key={index}><span>{featureLabel(change.feature)}</span><b>{formatMoneyIfNeeded(change.feature, change.from_value)}</b><ArrowRight size={14} /><b className="positive-text">{formatMoneyIfNeeded(change.feature, change.to_value)}</b></div>)}</div><div className="cf-footer"><span>Change cost: {cf.change_cost.toFixed(3)}</span><span>Model simulation only</span></div></div>;
}

function formatMoneyIfNeeded(feature, value) {
  return ['Applicant_Income', 'Coapplicant_Income', 'Loan_Amount'].includes(feature) ? formatMoney(value) : formatValue(value);
}

function EmptyState({ title, text, action, onClick }) {
  return <Layout section="LoanLens"><div className="empty-state"><div className="info-icon"><Activity size={22} /></div><h1>{title}</h1><p>{text}</p><button className="btn primary" onClick={onClick}>{action} <ArrowRight size={17} /></button></div></Layout>;
}

function downloadJSON(data) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `LoanLens_Analysis_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

function App() {
  return <Routes>
    <Route path="/" element={<Landing />} />
    <Route path="/home" element={<Home />} />
    <Route path="/apply" element={<Apply />} />
    <Route path="/decision" element={<Decision />} />
    <Route path="/what-if" element={<WhatIf />} />
    <Route path="*" element={<Navigate to="/" replace />} />
  </Routes>;
}

export default App;
