import ReferenceApp from "./ReferenceApp";

export default function App() {
  if (import.meta.env.VITE_DEMO_MODE === "true") {
    return (
      <div className="demo-mode">
        <ReferenceApp />
        <DemoBanner />
      </div>
    );
  }
  return <ReferenceApp />;
}

import DemoBanner from "./DemoBanner";
