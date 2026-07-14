import React from "react";
import { createRoot } from "react-dom/client";
import { GeometryWorkspace } from "../app/GeometryWorkspace";
import "../app/globals.css";

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <GeometryWorkspace />
  </React.StrictMode>,
);
