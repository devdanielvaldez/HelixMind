const MESSAGES = {
  en: {
    language: "Language",
    workspace: "WORKSPACE",
    overview: "Overview",
    literature: "Literature",
    biology: "Biology",
    graph: "Knowledge graph",
    hypotheses: "Hypotheses",
    reproducibility: "Reproducibility",
    capabilities: "Capabilities",
    review: "Review",
    credentials: "Credentials",
    titleOverview: "Research workspace",
    titleLiterature: "Literature explorer",
    titleBiology: "Biomedical entities",
    titleGraph: "Knowledge graph",
    titleHypotheses: "Hypothesis review",
    titleReproducibility: "Reproducibility record",
    titleCapabilities: "Capability registry",
    titleReview: "Human review",
    titleCredentials: "Credential administration",
    localSystem: "Local system",
    signIn: "Sign in",
    signingIn: "Signing in…",
    signOut: "Sign out",
    loginHeading: "Sign in to your research workspace",
    loginDescription: "Use your personal HelixMind API credential. The server validates its role and tenant, then keeps it in an encrypted, HttpOnly session cookie.",
    apiCredential: "Personal API credential",
    researchDirector: "RESEARCH DIRECTOR",
    heroHeadline: "Turn a research question into an evidence trail.",
    heroDescription: "Create a persistent research session and inspect its dependency-aware investigation plan.",
    objective: "Research objective",
    objectivePlaceholder: "What would you like to investigate?",
    depth: "Research depth",
    standard: "Standard",
    deep: "Deep",
    creditBudget: "Execution credit budget",
    investigate: "Investigate",
    researchSessions: "Research sessions",
    persistentInvestigations: "Persistent computational investigations",
    noSessions: "No research sessions yet",
    startWith: "Start with a disease, mechanism, gene, or scientific question.",
    signedInAs: "Signed in as",
    statResearchSessions: "Research sessions",
    statPersistentInvestigations: "persistent investigations",
    statKnowledgeGraph: "Knowledge graph",
    statSourcedRelationships: "sourced relationships",
    statEvidenceRecords: "Evidence records",
    statSourceLinkedClaims: "source-linked claims",
    statResearchArtifacts: "Research artifacts",
    statHashedSources: "content-hashed source records",
    statCapabilities: "Capabilities",
    statGovernedEntries: "governed registry entries",
    semanticPairsPending: "Semantic result pairs still awaiting screening",
    resumeSemanticPairs: "Resume remaining comparisons",
    alternativeRoutesIncomplete: "Alternative-route search reached its limit; additional routes may remain unexamined.",
  },
  es: {
    language: "Idioma",
    workspace: "ESPACIO DE TRABAJO",
    overview: "Resumen",
    literature: "Literatura",
    biology: "Biología",
    graph: "Grafo de conocimiento",
    hypotheses: "Hipótesis",
    reproducibility: "Reproducibilidad",
    capabilities: "Capacidades",
    review: "Revisión",
    credentials: "Credenciales",
    titleOverview: "Espacio de investigación",
    titleLiterature: "Explorador de literatura",
    titleBiology: "Entidades biomédicas",
    titleGraph: "Grafo de conocimiento",
    titleHypotheses: "Revisión de hipótesis",
    titleReproducibility: "Registro de reproducibilidad",
    titleCapabilities: "Registro de capacidades",
    titleReview: "Revisión humana",
    titleCredentials: "Administración de credenciales",
    localSystem: "Sistema local",
    signIn: "Iniciar sesión",
    signingIn: "Iniciando sesión…",
    signOut: "Cerrar sesión",
    loginHeading: "Entra a tu espacio de investigación",
    loginDescription: "Usa tu credencial personal de la API HelixMind. El servidor valida el rol y el tenant, y la guarda en una cookie de sesión cifrada y HttpOnly.",
    apiCredential: "Credencial personal de API",
    researchDirector: "DIRECCIÓN DE INVESTIGACIÓN",
    heroHeadline: "Convierte una pregunta científica en un recorrido de evidencia.",
    heroDescription: "Crea una sesión de investigación persistente y consulta su plan con dependencias.",
    objective: "Objetivo de investigación",
    objectivePlaceholder: "¿Qué te gustaría investigar?",
    depth: "Profundidad de investigación",
    standard: "Estándar",
    deep: "Profunda",
    creditBudget: "Presupuesto de créditos de ejecución",
    investigate: "Investigar",
    researchSessions: "Sesiones de investigación",
    persistentInvestigations: "Investigaciones computacionales persistentes",
    noSessions: "Aún no hay sesiones de investigación",
    startWith: "Empieza con una enfermedad, mecanismo, gen o pregunta científica.",
    signedInAs: "Sesión de",
    statResearchSessions: "Sesiones de investigación",
    statPersistentInvestigations: "investigaciones persistentes",
    statKnowledgeGraph: "Grafo de conocimiento",
    statSourcedRelationships: "relaciones con fuentes",
    statEvidenceRecords: "Registros de evidencia",
    statSourceLinkedClaims: "afirmaciones vinculadas a fuentes",
    statResearchArtifacts: "Artefactos de investigación",
    statHashedSources: "registros fuente con hash de contenido",
    statCapabilities: "Capacidades",
    statGovernedEntries: "entradas del registro gobernado",
    semanticPairsPending: "Pares de resultados que aún esperan análisis",
    resumeSemanticPairs: "Reanudar las comparaciones pendientes",
    alternativeRoutesIncomplete: "La búsqueda de rutas alternativas alcanzó su límite; podrían quedar rutas sin revisar.",
  },
};

export function normalizeLanguage(value) {
  return typeof value === "string" && value.toLowerCase().startsWith("es") ? "es" : "en";
}

export function translate(language, key) {
  const locale = normalizeLanguage(language);
  return MESSAGES[locale][key] ?? MESSAGES.en[key] ?? key;
}

export function readLanguage(storage, browserLanguage) {
  try {
    const saved = storage?.getItem("medai-language");
    if (saved === "en" || saved === "es") return saved;
  } catch { /* Browser storage may be unavailable; use the browser locale. */ }
  return normalizeLanguage(browserLanguage);
}

export function saveLanguage(storage, language) {
  const locale = normalizeLanguage(language);
  try { storage?.setItem("medai-language", locale); } catch { /* Preference remains active for this page. */ }
  return locale;
}
