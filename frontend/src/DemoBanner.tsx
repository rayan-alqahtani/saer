export default function DemoBanner() {
  return (
    <div
      className="demo-disclosure-banner"
      dir="rtl"
      role="note"
      style={{
        position: "fixed",
        left: 10,
        bottom: 10,
        zIndex: 10000,
        pointerEvents: "none",
        padding: "5px 9px",
        border: "1px solid rgba(0, 190, 255, .55)",
        borderRadius: 8,
        background: "rgba(3, 20, 43, .86)",
        color: "#dff7ff",
        font: "600 11px/1.4 Cairo, sans-serif",
        boxShadow: "0 0 12px rgba(0, 164, 255, .16)",
        whiteSpace: "nowrap",
      }}
    >
      عرض تجريبي: بيانات تاريخية؛ تقدير السعر يتطلب خادم النماذج
    </div>
  );
}
