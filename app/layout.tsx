import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "几何图校对器",
  description: "在设备本地识别印刷二维几何图中的点、线、圆和标签候选，并人工校正连接关系。",
  manifest: "/manifest.webmanifest",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
