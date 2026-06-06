import "./globals.css";

export const metadata = {
  title: "KRONECTOR F1 Intelligence",
  description: "Predict F1 race outcomes using natural language queries powered by Hugging Face and MLflow.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
