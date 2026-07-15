import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI NeuroOnco — Research Evaluation Platform",
  description:
    "Research-only evaluation platform for reproducible brain-tumor MRI classification, calibration, explainability, and literature retrieval experiments.",
  keywords: ["brain tumor", "MRI classification", "neuro-oncology", "AI", "Grad-CAM", "RAG", "MedGemma"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
