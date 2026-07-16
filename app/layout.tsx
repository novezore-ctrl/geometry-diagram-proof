import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "几何图校对器",
  description: "在设备本地校正几何题配图，把题干、必画线清单、共线关系和已确认元素一起分享给 GPT。",
  manifest: "/manifest.webmanifest",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
