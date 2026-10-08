import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./style.css";
import "@xyflow/react/dist/style.css";
export const metadata: Metadata = { title: "MedAI Research", description: "Computational biomedical research workspace" };
export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) { return <html lang="en"><body>{children}</body></html>; }
