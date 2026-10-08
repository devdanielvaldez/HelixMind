/** Build a ClinicalTrials.gov history URL only for a syntactically valid NCT ID. */
export function officialTrialHistoryUrl(value) {
  if (typeof value !== "string" || !/^NCT\d{8}$/.test(value)) return null;
  return `https://clinicaltrials.gov/study/${value}?tab=history`;
}
