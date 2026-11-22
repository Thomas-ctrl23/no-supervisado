import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Detección de Fraude en Tiempo Real | DBSCAN No Supervisado",
  description: "Sistema de seguridad bancaria impulsado por clustering DBSCAN y reducción con PCA",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="es">
      <body className="min-h-screen antialiased bg-slate-950 text-slate-100">
        {children}
      </body>
    </html>
  );
}

