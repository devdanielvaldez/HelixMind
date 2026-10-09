export type Language = "en" | "es";
export type MessageKey =
  | "language" | "workspace" | "overview" | "literature" | "biology" | "graph"
  | "hypotheses" | "reproducibility" | "capabilities" | "review" | "credentials"
  | "titleOverview" | "titleLiterature" | "titleBiology" | "titleGraph" | "titleHypotheses"
  | "titleReproducibility" | "titleCapabilities" | "titleReview" | "titleCredentials"
  | "localSystem" | "signIn" | "signingIn" | "signOut" | "loginHeading" | "loginDescription"
  | "apiCredential" | "researchDirector" | "heroHeadline" | "heroDescription" | "objective"
  | "objectivePlaceholder" | "depth" | "standard" | "deep" | "creditBudget" | "investigate"
  | "researchSessions" | "persistentInvestigations" | "noSessions" | "startWith" | "signedInAs"
  | "statResearchSessions" | "statPersistentInvestigations" | "statKnowledgeGraph"
  | "statSourcedRelationships" | "statEvidenceRecords" | "statSourceLinkedClaims"
  | "statResearchArtifacts" | "statHashedSources" | "statCapabilities" | "statGovernedEntries"
  | "semanticPairsPending" | "resumeSemanticPairs" | "alternativeRoutesIncomplete";
export function normalizeLanguage(value: unknown): Language;
export function translate(language: unknown, key: MessageKey): string;
export function readLanguage(storage: Pick<Storage, "getItem"> | null | undefined, browserLanguage: unknown): Language;
export function saveLanguage(storage: Pick<Storage, "setItem"> | null | undefined, language: unknown): Language;
