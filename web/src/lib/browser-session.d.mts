export const COOKIE_NAME: string;
export const SESSION_MAX_AGE_SECONDS: number;
export type BrowserSession = {
  token: string;
  subject: string;
  tenant_id: string;
  role: "reader" | "researcher" | "tenant_admin" | "platform_admin" | "service";
  expires_at: number;
};
export type BrowserIdentity = Pick<BrowserSession, "subject" | "tenant_id" | "role">;
export function isValidBrowserIdentity(value: unknown): value is BrowserIdentity;
export function encryptBrowserSession(session: BrowserSession, secret: string, now?: number): string;
export function decryptBrowserSession(value: string | undefined, secret: string, now?: number): BrowserSession | null;
