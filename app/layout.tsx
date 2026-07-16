import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "几何图校对器",
  description: "在设备本地校正几何题配图，把原题图片和人工校正后的图形结构说明分享给 GPT，由 GPT 直接从原图读取题干。",
  manifest: "/manifest.webmanifest",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
