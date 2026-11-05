import {
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type FormEvent,
  type ReactNode,
} from "react";
import LoginPage from "./LoginPage";
import { APP_BRAND_MARK, APP_LOGO, APP_NAME } from "./config";
import { hamoudEmptyResultHtml, hamoudFlowerHtml, hamoudHeroHtml } from "./hamoud-live-original";
import { prepareImage } from "./services/imageUpload";
import { createPortal } from "react-dom";
import layouts from "./data/reference-layout.json";
import {
  uiService,
  ar,
  arModel,
  number,
  vehicleName,
  type Listing,
  type Filters,
  type DatasetMeta,
  type MarketData,
} from "./services/uiService";
type Page = keyof typeof layouts;
type Rect = { x: number; y: number; w: number; h: number };
type Part = Rect & { src: string };
const box = (r: Rect): CSSProperties => ({
  left: r.x,
  top: r.y,
  width: r.w,
  height: r.h,
});
const part = (page: Page, key: string): Part =>
  (layouts[page].parts as unknown as Record<string, Part>)[key];
const paths: Record<string, string> = {
  home: "/",
  sell: "/sell",
  details: "/sell/details",
  estimate: "/estimate-from-image",
  budget: "/budget",
  compare: "/compare",
  login: "/login",
  market: "/market",
  browse: "/listings",
};
const navItems = [
  { label: "الرئيسية", href: "/" },
  { label: "خدماتنا", href: "/sell" },
  { label: "مقارنة السيارات", href: "/compare" },
  { label: "السوق", href: "/market" },
  { label: "عروض السيارات", href: "/listings" },
];
const titles: Record<string, string> = {
  home: `مرحباً بك في ${APP_NAME}`,
  sell: "بكم أبيع سيارتي؟",
  details: "أدخل بيانات سيارتك",
  estimate: "تقدير سعر السيارة من الصورة",
  budget: "اعثر على سيارتك المناسبة لميزانيتك",
  compare: "قارن بين سيارتين",
  login: "تسجيل الدخول",
  market: "تحليل السوق",
  browse: "استعرض إعلانات السيارات",
};
const route = () =>
  Object.keys(paths).find((key) => paths[key] === location.pathname) || "home";
function guardedRoute() {
  const current = route();
  if (import.meta.env.VITE_DEMO_MODE !== "true" || current === "login") return current;
  let hasGuestSession = false;
  try {
    hasGuestSession = sessionStorage.getItem("saer_guest") === "true" && Boolean(sessionStorage.getItem("saer_guest_name"));
  } catch {
    hasGuestSession = false;
  }
  if (hasGuestSession) return current;
  history.replaceState({}, "", "/login");
  return "login";
}
const go = (path: string) => {
  history.pushState({}, "", path);
  window.dispatchEvent(new PopStateEvent("popstate"));
  window.scrollTo(0, 0);
};

function Frame({ page, children }: { page: Page; children: ReactNode }) {
  const host = useRef<HTMLDivElement>(null);
  const [scale, setScale] = useState(1);
  const size = layouts[page];
  useEffect(() => {
    const el = host.current!;
    const resize = () => setScale(el.clientWidth / size.width);
    resize();
    const obs = new ResizeObserver(resize);
    obs.observe(el);
    return () => obs.disconnect();
  }, [size.width]);
  return (
    <>
    <div className="supplemental-nav"><Navbar page={page} overlay /></div>
    <div
      ref={host}
      className="frame-host"
      style={{ height: size.height * scale }}
    >
      <main
        className={`reference-frame page-${page}`}
        aria-label={titles[page]}
        dir="rtl"
        style={{
          width: size.width,
          height: size.height,
          transform: `scale(${scale})`,
        }}
      >
        <img
          className="scene-background"
          src={size.background}
          alt=""
          draggable={false}
        />
        {children}
      </main>
    </div>
    </>
  );
}
function GlassCard({
  page,
  id,
  children,
  label,
  className = "",
  artSrc,
}: {
  page: Page;
  id: string;
  children?: ReactNode;
  label?: string;
  className?: string;
  artSrc?: string;
}) {
  const p = part(page, id);
  return (
    <section
      className={`art-component glass-card ${className}`}
      style={{ ...box(p), backgroundImage: `url("${artSrc || p.src}")` }}
      aria-label={label}
    >
      {children}
    </section>
  );
}
function StaticArt({
  page,
  id,
  text,
  heading = false,
}: {
  page: Page;
  id: string;
  text: string;
  heading?: boolean;
}) {
  const p = part(page, id);
  return (
    <div className="art-component" style={box(p)}>
      <img src={p.src} alt="" draggable={false} />
      {heading ? (
        <h1 className="sr-only">{text}</h1>
      ) : (
        <span className="sr-only">{text}</span>
      )}
    </div>
  );
}
function Action({
  at,
  label,
  onClick,
  href,
  children,
  type = "button",
  className = "",
  disabled = false,
}: {
  at: Rect;
  label: string;
  onClick?: () => void;
  href?: string;
  children?: ReactNode;
  type?: "button" | "submit";
  className?: string;
  disabled?: boolean;
}) {
  const props = {
    className: `image-action ${className}`,
    style: box(at),
    "aria-label": label,
  };
  return href ? (
    <a
      {...props}
      href={href}
      onClick={(e) => {
        e.preventDefault();
        go(href);
      }}
    >
      <span className="sr-only">{label}</span>
      {children}
    </a>
  ) : (
    <button {...props} type={type} onClick={onClick} disabled={disabled}>
      <span className="sr-only">{label}</span>
      {children}
    </button>
  );
}
function PrimaryButton({
  page,
  id,
  label,
  onClick,
  disabled = false,
}: {
  page: Page;
  id: string;
  label: string;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <GlassCard page={page} id={id}>
      <Action
        at={{ x: 0, y: 0, w: part(page, id).w, h: part(page, id).h }}
        label={label}
        onClick={onClick}
        disabled={disabled}
      />
    </GlassCard>
  );
}
function Navbar({ page, home = false, overlay = false }: { page: Page | "market" | "browse"; home?: boolean; overlay?: boolean }) {
  const [viewportScale, setViewportScale] = useState(() => Math.min(1, window.innerWidth / 1536));
  const [guestName, setGuestName] = useState(() => {
    try { return sessionStorage.getItem("saer_guest_name") || ""; } catch { return ""; }
  });
  const [guestMenuOpen, setGuestMenuOpen] = useState(false);
  const [editingGuestName, setEditingGuestName] = useState(false);
  const [editedName, setEditedName] = useState(guestName);
  const guestAccount = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const sync = () => {
      try { setGuestName(sessionStorage.getItem("saer_guest_name") || ""); } catch { setGuestName(""); }
    };
    window.addEventListener("saer-guest-name", sync);
    window.addEventListener("storage", sync);
    return () => { window.removeEventListener("saer-guest-name", sync); window.removeEventListener("storage", sync); };
  }, []);
  useEffect(() => {
    if (!guestMenuOpen && !editingGuestName) return;
    const onPointerDown = (event: PointerEvent) => {
      if (guestAccount.current && !guestAccount.current.contains(event.target as Node)) setGuestMenuOpen(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") { setGuestMenuOpen(false); setEditingGuestName(false); }
    };
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => { document.removeEventListener("pointerdown", onPointerDown); document.removeEventListener("keydown", onKeyDown); };
  }, [guestMenuOpen, editingGuestName]);
  const saveGuestName = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const firstName = editedName.trim().split(/\s+/)[0];
    if (!firstName) return;
    sessionStorage.setItem("saer_guest_name", firstName);
    window.dispatchEvent(new Event("saer-guest-name"));
    setGuestName(firstName);
    setEditingGuestName(false);
    setGuestMenuOpen(false);
  };
  const logoutGuest = () => {
    sessionStorage.removeItem("saer_guest");
    sessionStorage.removeItem("saer_guest_name");
    window.dispatchEvent(new Event("saer-guest-name"));
    go("/login");
  };
  useEffect(() => {
    const resize = () => setViewportScale(Math.min(1, window.innerWidth / 1536));
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, []);
  const currentPath = location.pathname;
  const frameHeader = !home && !overlay && page !== "market" && page !== "browse";
  const headerStyle: CSSProperties | undefined = overlay ? {
    position: "relative", transform: "none",
  } : frameHeader ? {
    left: `${16 / viewportScale}px`, top: `${14 / viewportScale}px`,
    width: `${window.innerWidth - 32}px`, height: "80px",
    transform: `scale(${1 / viewportScale})`, transformOrigin: "top left",
  } : undefined;
  return (
    <header className={`site-header ${home ? "site-header-home" : ""} ${overlay ? "site-header-overlay" : ""}`} style={headerStyle} aria-label="رأس الموقع">
      <a className="site-brand" href="/" onClick={event => { event.preventDefault(); go("/"); }} aria-label={APP_NAME}>
        <img src={APP_BRAND_MARK} alt={APP_NAME} />
      </a>
      <nav className="site-nav" aria-label="التنقل الرئيسي">
        {navItems.map(item => {
          const active = item.href === "/" ? currentPath === "/" || currentPath === "/login" : currentPath === item.href || (item.href === "/sell" && ["/sell/details", "/budget"].includes(currentPath));
          return <a key={item.href} href={item.href} className={active ? "active" : ""} aria-current={active ? "page" : undefined} onClick={event => { event.preventDefault(); go(item.href); }}>{item.label}</a>;
        })}
      </nav>
      <div className="site-account-wrap" ref={guestAccount}>
        {guestName ? <button className="site-account" type="button" aria-haspopup="menu" aria-expanded={guestMenuOpen} onClick={() => setGuestMenuOpen(open => !open)}>
          <span className="site-avatar" aria-hidden="true">{guestName.slice(0, 1)}</span><span>{guestName}</span><span className="site-account-caret" aria-hidden="true">⌄</span>
        </button> : <div className="site-account"><span className="site-avatar" aria-hidden="true">ز</span><span>زائر</span></div>}
        {guestMenuOpen && <div className="guest-account-menu" role="menu">
          <button type="button" role="menuitem" onClick={() => { setEditedName(guestName); setGuestMenuOpen(false); setEditingGuestName(true); }}>تعديل الاسم</button>
          <button type="button" role="menuitem" onClick={logoutGuest}>تسجيل الخروج</button>
        </div>}
        {editingGuestName && createPortal(<div className="guest-name-backdrop" role="presentation" onMouseDown={event => { if (event.target === event.currentTarget) setEditingGuestName(false); }}>
          <form className="guest-name-dialog" role="dialog" aria-modal="true" aria-labelledby="edit-guest-name-title" onSubmit={saveGuestName}>
            <h1 id="edit-guest-name-title">تعديل الاسم</h1>
            <label htmlFor="edit-guest-first-name">الاسم الأول</label>
            <input id="edit-guest-first-name" autoFocus required maxLength={40} value={editedName} onChange={event => setEditedName(event.target.value)} placeholder="اكتب اسمك الأول" />
            <button type="submit">حفظ الاسم</button>
            <button className="guest-name-cancel" type="button" onClick={() => setEditingGuestName(false)}>إلغاء</button>
          </form>
        </div>, document.body)}
      </div>
    </header>
  );
}
function Field({
  at,
  label,
  value,
  onChange,
  choices,
  type = "text",
  required = false,
}: {
  at: Rect;
  label: string;
  value: string;
  onChange: (v: string) => void;
  choices?: string[];
  type?: string;
  required?: boolean;
}) {
  const props = {
    className: `native-field ${value ? "has-value" : ""}`,
    style: box(at),
    "aria-label": label,
    required,
    value,
  };
  return choices ? (
    <select {...props} onChange={(e) => onChange(e.target.value)}>
      <option value="">{label}</option>
      {choices.map((x) => (
        <option key={x} value={x}>
          {arModel(x) !== x ? `${arModel(x)} (${x})` : ar(x)}
        </option>
      ))}
    </select>
  ) : (
    <input
      {...props}
      type={type}
      min={type === "number" ? 0 : undefined}
      onChange={(e) => onChange(e.target.value)}
    />
  );
}
function Crumb({ page, text }: { page: Page; text: string }) {
  return (
    <GlassCard page={page} id="breadcrumb" label={text}>
      <Action
        at={{
          x: Math.max(0, part(page, "breadcrumb").w - 128),
          y: 0,
          w: 128,
          h: 40,
        }}
        label="الرئيسية"
        href="/"
      />
    </GlassCard>
  );
}
function Home() {
  const links = [
    {label:"أفضل 5 سيارات حسب ميزانيتك",href:"/budget",x:3.1,y:28.1,w:30.8,h:23.0},
    {label:"بكم أبيع سيارتي؟",href:"/sell",x:35.1,y:28.1,w:30.4,h:23.0},
    {label:"تقدير سعر السيارة من الصورة",href:"/estimate-from-image",x:66.6,y:28.1,w:30.6,h:23.0},
    {label:"استعرض إعلانات السيارات",href:"/listings",x:3.1,y:52.4,w:30.8,h:23.8},
    {label:"تحليل السوق",href:"/market",x:35.1,y:52.4,w:30.4,h:23.8},
    {label:"قارن بين سيارتين",href:"/compare",x:66.6,y:52.4,w:30.6,h:23.8},
    {label:"إعلانات تويوتا",href:"/listings?make=Toyota",x:37.7,y:25.1,w:6.4,h:4.2},
    {label:"إعلانات هيونداي",href:"/listings?make=Hyundai",x:44.1,y:25.1,w:6.4,h:4.2},
    {label:"إعلانات هوندا",href:"/listings?make=Honda",x:50.5,y:25.1,w:6.4,h:4.2},
    {label:"إعلانات شيفروليه",href:"/listings?make=Chevrolet",x:56.9,y:25.1,w:6.4,h:4.2},
  ];
  return (
    <main className="home-screenshot-page" dir="rtl" aria-label={`الصفحة الرئيسية في ${APP_NAME}`}>
      <img src="/ui/home/home-final.png" alt="الصفحة الرئيسية لسعّر" draggable={false}/>
      <div className="home-budget-cars-panel" aria-label="السيارات الخمس في بطاقة أفضل 5 سيارات حسب ميزانيتك">
        <img src="/ui/home/budget-cars-cutout.png" alt="السيارات الخمس" draggable={false}/>
      </div>
      <div className="home-sell-car-panel" aria-label="السيارة في بطاقة بكم أبيع سيارتي؟">
        <img src="/ui/home/sell-car-cutout.png" alt="سيارة نيسان فيرليدي سوداء" draggable={false}/>
      </div>
      <img className="home-compare-feature-user" src="/ui/home/compare-feature-user.png" alt="قارن بين سيارتين" draggable={false}/>
      <img className="home-browse-feature-user" src="/ui/home/browse-feature-user.png" alt="استعرض إعلانات السيارات" draggable={false}/>
      <img className="home-estimate-feature-user" src="/ui/home/estimate-feature-user.png" alt="تقدير سعر السيارة من الصورة" draggable={false}/>
      <Navbar page="home" home />
      {links.map(({label,href,x,y,w,h})=>(
        <a key={label} className="home-screenshot-hotspot" href={href} aria-label={label}
          style={{left:`${x}%`,top:`${y}%`,width:`${w}%`,height:`${h}%`}}
          onClick={event=>{event.preventDefault();go(href);}} />
      ))}
    </main>
  );
}
function Sell({
  onUpload,
  onManual,
}: {
  onUpload: () => void;
  onManual: () => void;
}) {
  return (
    <Frame page="sell">
      <StaticArt
        page="sell"
        id="heading"
        text="بكم أبيع سيارتي؟ اختر طريقة إدخال بياناتك"
        heading
      />
      <GlassCard page="sell" id="photo">
        <Action
          at={{ x: 0, y: 0, w: 600, h: 496 }}
          label="ارفع صورة السيارة"
          onClick={onUpload}
        >
          <span className="sell-photo-visible-title" aria-hidden="true">ارفع صورة السيارة</span>
        </Action>
      </GlassCard>
      <GlassCard page="sell" id="manual">
        <Action
          at={{ x: 0, y: 0, w: 603, h: 496 }}
          label="اختر من القائمة"
          onClick={onManual}
        />
      </GlassCard>
      <StaticArt
        page="sell"
        id="information"
        text="بعد اختيار طريقة إدخال سيارتك، ستقوم بإدخال سنة الصنع، عدد الكيلومترات وتفاصيل المركبة قبل الحصول على تقدير السعر."
      />
    </Frame>
  );
}
function Feedback({text, close}: {text:string; close:()=>void}) { return text ? createPortal(<div className="surface-feedback" role="status"><span>{text}</span><button onClick={close} aria-label="إغلاق التنبيه">×</button></div>, document.body) : null; }
function FallingNightFlowers() {
  const [visible, setVisible] = useState(true);
  const [closing, setClosing] = useState(false);
  useEffect(() => {
    const closeTimer = window.setTimeout(() => setClosing(true), 2600);
    const removeTimer = window.setTimeout(() => setVisible(false), 3100);
    return () => {
      window.clearTimeout(closeTimer);
      window.clearTimeout(removeTimer);
    };
  }, []);
  if (!visible) return null;

  const flowerSvg = hamoudFlowerHtml
    .replace('id="p0"', 'id="loading-flower-gradient"')
    .replaceAll("url(#p0)", "url(#loading-flower-gradient)")
    .replace('width="36"', 'width="100%"')
    .replace(' style="animation-delay:0s"', "");
  const flowerImage = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(flowerSvg)}`;
  return (
    <div className={`falling-night-flowers ${closing ? "is-closing" : ""}`} role="status" aria-label="جارٍ تجهيز صفحة تقدير السعر">
      <div className="falling-night-flower-field" aria-hidden="true">
      {Array.from({ length: 44 }, (_, index) => {
        const style = {
          left: `${(index * 37 + 3) % 100}%`,
          width: `${20 + ((index * 13) % 22)}px`,
          top: `${-12 + ((index * 47) % 105)}vh`,
          animation: `saer-flower-rain ${2.5 + ((index * 7) % 15) / 10}s ease-in ${((index * 11) % 8) / 10}s both`,
          "--flower-drift": `${((index * 19) % 151) - 75}px`,
        } as CSSProperties;
        return <span className="falling-night-flower" key={index} style={style}><img src={flowerImage} alt="" /></span>;
      })}
      </div>
      <div className="falling-night-loading" aria-hidden="true">
        <strong>{APP_NAME}</strong>
        <span>جارٍ تجهيز تقدير السعر</span>
        <i><b /></i>
      </div>
    </div>
  );
}
const roadCars = [
  { name: "السيارة الأولى", src: "/assets/road-cars/car1.png", className: "road-vehicle--car1" },
  { name: "السيارة الثانية", src: "/assets/road-cars/car2.png", className: "road-vehicle--car2" },
  { name: "السيارة الثالثة", src: "/assets/road-cars/car3.png", className: "road-vehicle--car3" },
  { name: "السيارة الرابعة", src: "/assets/road-cars/car4.png", className: "road-vehicle--car4" },
  { name: "السيارة الخامسة", src: "/assets/road-cars/car5.png", className: "road-vehicle--car5" },
];
function estimateHeroWithRoadCars() {
  const vehicleSet = (clone = false) => `<div class="road-vehicle-set${clone ? " road-vehicle-set--clone" : ""}"${clone ? " aria-hidden=\"true\"" : ""}>${roadCars.map((car) => `<span class="road-vehicle-item ${car.className}"><i class="road-vehicle-beam" aria-hidden="true"></i><img src="${car.src}" alt="" loading="lazy" decoding="async" /></span>`).join("")}</div>`;
  const track = `<div class="road-traffic-track" aria-hidden="true">${vehicleSet()}${vehicleSet(true)}</div>`;
  const cleanHero = hamoudHeroHtml;
  const roadStart = cleanHero.indexOf('<div class="road">');
  if (roadStart < 0) return cleanHero;
  const roadClose = cleanHero.indexOf("</div>", roadStart);
  if (roadClose < 0) return cleanHero;
  return `${cleanHero.slice(0, roadClose)}${track}${cleanHero.slice(roadClose)}`;
}
function Estimate() {
  const input = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<File[]>([]),
    [previews, setPreviews] = useState<string[]>([]),
    [status, setStatus] = useState(""),
    [imagePrice, setImagePrice] = useState<{price:number;lower80?:number;upper80?:number;lower90?:number;upper90?:number;imageCount:number;notice:string}|null>(null),
    [estimating, setEstimating] = useState(false),
    [nightMode, setNightMode] = useState(true);
  const emptyResultHtml = hamoudEmptyResultHtml
    .replace(/<svg viewBox="0 0 140 56"[\s\S]*?<\/svg>/, '<img class="empty-car-photo" src="/assets/road-cars/car1.png" alt="" loading="lazy" decoding="async" />')
    .replace("ارفع صور سيارة (من 1 إلى 8)", "ارفع صورة السيارة")
    .replace("أو جرّب «عيّنة عشوائية» من عيّنات الاختبار.", "");
  const estimateHeroHtml = estimateHeroWithRoadCars().replace("سعّر", APP_NAME);
  useEffect(() => () => previews.forEach((url) => URL.revokeObjectURL(url)), [previews]);
  const selectionId = useRef(0);
  useEffect(() => () => { selectionId.current++; }, []);
  const choose = async (selected: FileList | null) => {
    if (!selected?.length) return;
    if (selected.length !== 1) { setStatus("ارفع صورة واحدة فقط للسيارة."); if (input.current) input.current.value = ""; return; }
    const file = selected[0];
    const id = ++selectionId.current;
    setImagePrice(null); setStatus("");
    try {
      const preview = await prepareImage(file);
      if (id !== selectionId.current) { URL.revokeObjectURL(preview); return; }
      setFiles([file]); setPreviews([preview]);
      setStatus("");
    } catch (error) {
      if (id === selectionId.current) {
        setFiles([]); setPreviews([]);
        setStatus(error instanceof Error ? error.message : "تعذر فتح الصورة.");
      }
    } finally { if (id === selectionId.current && input.current) input.current.value = ""; }
  };
  const clear = () => { selectionId.current++; setFiles([]); setPreviews([]); setImagePrice(null); setStatus(""); };
  return (
    <div className="hamoud-app">
    <nav className="hamoud-site-nav" dir="rtl" aria-label={`التنقل في ${APP_NAME}`}>
      <a onClick={event => { event.preventDefault(); go(event.currentTarget.getAttribute("href")!); }} className="hamoud-brand-link" href="/" aria-label={APP_NAME}><img src={APP_LOGO} alt={APP_NAME}/></a>
      <a onClick={event => { event.preventDefault(); go(event.currentTarget.getAttribute("href")!); }} href="/">الرئيسية</a><a onClick={event => { event.preventDefault(); go(event.currentTarget.getAttribute("href")!); }} href="/sell">بكم أبيع سيارتي؟</a><a onClick={event => { event.preventDefault(); go(event.currentTarget.getAttribute("href")!); }} className="active" href="/estimate-from-image">تقدير السعر من الصورة</a><a onClick={event => { event.preventDefault(); go(event.currentTarget.getAttribute("href")!); }} href="/budget">أفضل 5 سيارات</a><a onClick={event => { event.preventDefault(); go(event.currentTarget.getAttribute("href")!); }} href="/compare">مقارنة السيارات</a><a onClick={event => { event.preventDefault(); go(event.currentTarget.getAttribute("href")!); }} href="/market">تحليل السوق</a><a onClick={event => { event.preventDefault(); go(event.currentTarget.getAttribute("href")!); }} href="/listings">الإعلانات</a>
    </nav>
    <main className={`hamoud-live-page gradio-container ${nightMode ? "night" : ""}`} dir="rtl">
      <FallingNightFlowers />
      <button id="mode-btn" className="secondary" onClick={() => setNightMode((value) => !value)}>
        {nightMode ? "☀️ الوضع النهاري" : "🌙 الوضع الليلي"}
      </button>
      <div dangerouslySetInnerHTML={{ __html: estimateHeroHtml }} />
      <section className="hamoud-live-workspace" aria-label="تقدير سعر السيارة">
        <article className="hamoud-live-result" aria-live="polite">
          {imagePrice ? (
            <div className="card result">
              <div className="chips"><span className="chip">تقدير تجريبي</span></div>
              <div className="label">السعر المقدّر</div>
              <div className="price">{number(imagePrice.price)} <small>ريال سعودي</small></div>
              <div className="legend">
                <span><i className="dot d80" /> نطاق 80٪</span>
                <b>{imagePrice.lower80 != null && imagePrice.upper80 != null ? `${number(imagePrice.lower80)} – ${number(imagePrice.upper80)} ريال` : "غير متوفر"}</b>
                <span><i className="dot d90" /> نطاق 90٪</span>
                <b>{imagePrice.lower90 != null && imagePrice.upper90 != null ? `${number(imagePrice.lower90)} – ${number(imagePrice.upper90)} ريال` : "غير متوفر"}</b>
              </div>
              <p className="note">{imagePrice.notice}</p>
            </div>
          ) : (
            <div dangerouslySetInnerHTML={{ __html: emptyResultHtml }} />
          )}
        </article>
        <div className="hamoud-live-upload-column">
          <label
            className="hamoud-live-upload block"
            onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => { event.preventDefault(); choose(event.dataTransfer.files); }}
          >
            <span className="hamoud-live-count">▧ ارفع صورة السيارة</span>
            <input ref={input} type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => choose(event.target.files)} />
            {previews.length ? (
              <div className="hamoud-live-thumbs">{previews.map((src, index) => <img src={src} key={`${src}-${index}`} alt="صورة السيارة" />)}</div>
            ) : (
              <div className="hamoud-live-drop"><b aria-hidden="true">↥</b><span>أسقط صورة هنا</span><i>— أو —</i><span>انقر للتحميل</span></div>
            )}
          </label>
          {status && <p className="hamoud-live-error" role="status">{status}</p>}
          <button className="primary hamoud-live-estimate" onClick={async () => {
            if (!files.length) { setStatus("اختر صورة السيارة أولًا."); return; }
            const requestSelection = selectionId.current;
            setEstimating(true); setStatus(""); setImagePrice(null);
            try { const result = await uiService.estimateFromImage(files[0]); if (requestSelection === selectionId.current) setImagePrice({ ...result, imageCount: 1 }); }
            catch (error) { if (requestSelection === selectionId.current) setStatus(error instanceof Error ? error.message : "تعذر تقدير السعر."); }
            finally { setEstimating(false); }
          }} disabled={estimating}>{estimating ? "جارٍ التقدير…" : "🔍 قدّر السعر"}</button>
          <div className="hamoud-live-actions">
            <button className="secondary" onClick={clear}>🗑 حذف الصورة</button>
          </div>
        </div>
      </section>
    </main>
    </div>
  );
}
function Dialog({
  label,
  onClose,
  children,
}: {
  label: string;
  onClose: () => void;
  children: ReactNode;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    ref.current?.showModal();
  }, []);
  return (
    <dialog className="live-dialog" ref={ref} onCancel={onClose} dir="rtl">
      <div className="dialog-title">
        <h2>{label}</h2>
        <button aria-label="إغلاق" onClick={onClose}>
          ×
        </button>
      </div>
      {children}
    </dialog>
  );
}
function Details({ preview, manual, imageFile, meta, pricePairs, onReplacePhoto }: { preview: string; manual: boolean; imageFile: File | null; meta: DatasetMeta | null; pricePairs: Record<string, string[]>; onReplacePhoto: () => void }) {
  const [values, setValues] = useState<Record<string, string>>({
    year: "",
    mileage: "",
    trim: "",
    condition: "",
    make: "",
    model: "",
  });
  const [confirmed, setConfirmed] = useState(false),
    [picking, setPicking] = useState(manual),
    [reviewOpen, setReviewOpen] = useState(false),
    [identifying, setIdentifying] = useState(false),
    [identityNotice, setIdentityNotice] = useState(""),
    [suggestions, setSuggestions] = useState<{make:string;model:string;confidence:number}[]>([]),
    [salePrice, setSalePrice] = useState<{price:number;notice:string}|null>(null),
    [trims, setTrims] = useState<string[]>([]),
    [vehicleYears, setVehicleYears] = useState<number[]>([]),
    [trimsByYear, setTrimsByYear] = useState<Record<string,string[]>>({}),
    [trimLoading, setTrimLoading] = useState(false),
    [pricing, setPricing] = useState(false);
  useEffect(() => {
    if (manual || !imageFile) return;
    let active = true;
    setIdentifying(true);
    setIdentityNotice("");
    void uiService.identifyVehicle(imageFile).then((result) => {
      if (!active) return;
      setValues((v) => ({ ...v, make: result.make, model: result.model }));
      setSuggestions(result.alternatives);
      setConfirmed(false);
      setIdentityNotice("");
      setReviewOpen(Boolean(result.make && result.model));
    }).catch((error: unknown) => {
      if (active) {
        setReviewOpen(false);
        setIdentityNotice(error instanceof Error ? error.message : "تعذر التعرف على السيارة. اخترها يدويًا.");
      }
    }).finally(() => { if (active) setIdentifying(false); });
    return () => { active = false; };
  }, [imageFile, manual]);
  useEffect(() => {
    let active = true;
    setValues(v => ({ ...v, trim: "" }));
    if (!values.make || !values.model) { setTrims([]); setVehicleYears([]); setTrimsByYear({}); return; }
    setTrimLoading(true);
    void uiService.vehicleOptions(values.make, values.model)
      .then(result => { if (active) { setTrims(result.trims); setVehicleYears(result.years); setTrimsByYear(result.trimsByYear); } })
      .catch(() => { if (active) { setTrims([]); setVehicleYears([]); setTrimsByYear({}); } })
      .finally(() => { if (active) setTrimLoading(false); });
    return () => { active = false; };
  }, [values.make, values.model]);
  const trimChoices = values.year && trimsByYear[values.year]?.length ? trimsByYear[values.year] : trims;
  const set = (k: string) => (v: string) =>
    setValues((s) => ({
      ...s,
      [k]: v,
      ...(k === "make" ? { model: "", trim: "" } : {}),
      ...(k === "model" ? { trim: "" } : {}),
      ...(k === "year" ? { trim: "" } : {}),
    }));
  return (
    <Frame page="details">
      <StaticArt
        page="details"
        id="heading"
        text="أدخل بيانات سيارتك — أكمل البيانات لتحصل على تقدير سعر دقيق"
        heading
      />
      <GlassCard page="details" id="form" label="بيانات السيارة" artSrc="/ui/details/form-no-confirm.png">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (!values.make || !values.model) { setPicking(true); return; }
            if (!confirmed) { setIdentityNotice("أكّد صحة الماركة والموديل قبل حساب السعر."); return; }
            if (!values.year || values.mileage === "") { setIdentityNotice("سنة الصنع والممشى مطلوبان لحساب التقدير."); return; }
            if (!(pricePairs[values.make] || []).includes(values.model)) { setIdentityNotice("لا يتوفر تقدير سعري مدعوم لهذه الماركة والموديل في النموذج الحالي."); return; }
            setPricing(true);
            setIdentityNotice("");
            void uiService.estimateSale(values)
              .then(setSalePrice)
              .catch((error:unknown) => setIdentityNotice(error instanceof Error ? error.message : "تعذر تقدير السعر."))
              .finally(()=>setPricing(false));
          }}
        >
          {confirmed && <>
            <Field at={{ x: 45, y: 78, w: 361, h: 42 }} label="اختر سنة الصنع" value={values.year} onChange={set("year")} choices={(vehicleYears.length ? vehicleYears : meta?.years || []).map(String)} required />
            <Field at={{ x: 45, y: 136, w: 361, h: 41 }} label="أدخل الممشى (كم)" value={values.mileage} onChange={set("mileage")} type="number" required />
            <Field at={{ x: 45, y: 193, w: 361, h: 42 }} label={trimLoading ? "جارٍ تحميل الفئات" : "الفئة (اختياري)"} value={values.trim} onChange={set("trim")} choices={trimChoices} />
            <Field at={{ x: 45, y: 251, w: 361, h: 42 }} label="الحالة (اختياري)" value={values.condition} onChange={set("condition")} choices={["used", "new"]} />
          </>}
          <Action
            at={{ x: 44, y: 427, w: 548, h: 64 }}
            type="submit"
            label={pricing ? "جارٍ حساب السعر" : "احسب سعر الإعلان"}
            disabled={pricing}
          />
        </form>
        {!confirmed && <div className="details-form-lock">أكّد ماركة السيارة وموديلها أولًا لإكمال بيانات التسعير</div>}
        <div className="details-city-mask" aria-hidden="true" />
      </GlassCard>
      <GlassCard
        page="details"
        id="identity"
        label={manual ? "صورة السيارة" : "صورة سيارتك المرفوعة"}
        artSrc={manual ? "/ui/details/identity-manual.png" : undefined}
      >
        {preview && (
          <div
            className="uploaded-image"
            style={{ left: 42, top: 86, width: 478, height: 254 }}
          >
            <img src={preview} alt="صورة سيارتك المرفوعة" />
          </div>
        )}
        {preview && <Action at={{ x: 476, y: 82, w: 42, h: 42 }} className="photo-close" label="تغيير الصورة" children="×" onClick={onReplacePhoto} />}
        {values.make && (
          <div
            className="identity-value"
            style={{ left: 60, top: 354, width: 438, height: 48 }}
          >
            {vehicleName(values.make, values.model)}
          </div>
        )}
        {identifying && <div className="identity-feedback" role="status" style={{left:18,top:365,width:520}}>جارٍ تحليل الصورة…</div>}
        {values.make && !confirmed && !identifying && !reviewOpen && !manual && <button className="identity-review-reopen" type="button" onClick={() => setReviewOpen(true)}>مراجعة السيارة التي تعرّف عليها النظام</button>}
        {!values.make && !identifying && <button className="identity-manual" type="button" onClick={() => setPicking(true)}>اختيار الماركة والموديل يدويًا</button>}
        {confirmed && <div className="identity-feedback identity-confirmed" role="status" style={{left:18,top:410,width:520}}>تم تأكيد {vehicleName(values.make, values.model)}. أدخل سنة الصنع والممشى لإكمال التقدير.</div>}
        {identityNotice && <div className="identity-feedback" role="alert" style={{left:18,top:365,width:520}}>{identityNotice}</div>}
      </GlassCard>
      <StaticArt
        page="details"
        id="information"
        text="سيظهر التقدير بعد إدخال البيانات — السعر تقديري ويعتمد على البيانات المدخلة"
      />
      {picking && (
        <Dialog
          label={suggestions.length ? "اقتراح الماركة والموديل" : "اختر الماركة والموديل يدويًا"}
          onClose={() => setPicking(false)}
        >
          {suggestions.length > 0 && <div className="identity-suggestions"><p>اختر الماركة والموديل الصحيحين من القائمة</p></div>}
          <label>
            الماركة
            <select
              value={values.make}
              onChange={(e) => set("make")(e.target.value)}
            >
              <option value="">اختر الشركة</option>
              {(meta?.makes || Object.keys(pricePairs)).map((m) => (
                <option key={m} value={m}>{ar(m)} ({m})</option>
              ))}
            </select>
          </label>
          <label>
            الموديل
            <select
              value={values.model}
              onChange={(e) => set("model")(e.target.value)}
            >
              <option value="">اختر الموديل</option>
              {(meta?.models_by_make?.[values.make] || pricePairs[values.make] || []).map((model) => (
                <option key={model} value={model}>{arModel(model)}{arModel(model) !== model ? ` (${model})` : ""}</option>
              ))}
            </select>
          </label>
          <button
            className="live-primary"
            disabled={!values.make || !values.model}
            onClick={() => {
              setConfirmed(true);
              setPicking(false);
            }}
          >
            أؤكد صحة الماركة والموديل
          </button>
        </Dialog>
      )}
      {reviewOpen && !picking && !manual && imageFile && values.make && values.model && !confirmed && (
        <Dialog
          label="تأكيد السيارة التي تعرّف عليها النظام"
          onClose={() => setReviewOpen(false)}
        >
          <section className="identity-review" aria-label="مراجعة السيارة المتوقعة">
            <img src={preview} alt="صورة السيارة التي رفعتها" />
            <div className="identity-review-name">
              <span>توقع النظام</span>
              <strong>{vehicleName(values.make, values.model)}</strong>
              <p>هل هذه هي سيارتك؟</p>
            </div>
            <div className="identity-review-actions">
              <button className="identity-review-yes" type="button" onClick={() => { setConfirmed(true); setIdentityNotice(""); setReviewOpen(false); }}>✓ صحيح</button>
              <button className="identity-review-no" type="button" onClick={() => { setReviewOpen(false); setPicking(true); }}>✕ خطأ، اختر يدويًا</button>
            </div>
          </section>
        </Dialog>
      )}
      {salePrice && <Dialog label="تقدير سعر سيارتك" onClose={()=>setSalePrice(null)}><div className="price-result"><strong>{number(salePrice.price)} ريال سعودي</strong><p>{salePrice.notice}</p></div></Dialog>}
    </Frame>
  );
}
function ListingDetails({
  listing,
  onClose,
}: {
  listing: Listing;
  onClose: () => void;
}) {
  const [images, setImages] = useState<string[]>(listing.image ? [listing.image] : []),
    [loadingImages, setLoadingImages] = useState(true),
    [imageError, setImageError] = useState("");
  useEffect(() => {
    let active = true;
    setLoadingImages(true);
    setImageError("");
    void uiService.listingImages(listing.id).then((result) => {
      if (!active) return;
      const urls = result.images.map((image) => image.url);
      setImages(urls.length ? urls : listing.image ? [listing.image] : []);
    }).catch((error: unknown) => {
      if (!active) return;
      setImageError(error instanceof Error ? error.message : "تعذر تحميل صور الإعلان.");
      setImages(listing.image ? [listing.image] : []);
    }).finally(() => { if (active) setLoadingImages(false); });
    return () => { active = false; };
  }, [listing.id, listing.image]);
  return (
    <Dialog label={vehicleName(listing.make, listing.model)} onClose={onClose}>
      <section className="listing-photo-section" aria-label="صور الإعلان">
        <h3>صور الإعلان ({number(loadingImages ? listing.imageCount : images.length)})</h3>
        {loadingImages && <p className="history-note">جارٍ تحميل صور هذا الإعلان…</p>}
        {imageError && <p className="history-note">{imageError}</p>}
        {images.length ? <div className="listing-gallery">{images.map((src, index) =>
          <img key={`${listing.id}-${index}`} src={src} alt={`${vehicleName(listing.make, listing.model)} — الصورة ${number(index + 1)}`} loading="lazy" />
        )}</div> : !loadingImages && <p className="history-note">لا توجد صور محفوظة لهذا الإعلان.</p>}
      </section>
      <dl className="live-facts">
        {[
          ["سنة الصنع", String(listing.year ?? '—')],
          ["السعر المعلن", `${number(listing.price)} ريال`],
          ["الممشى", `${number(listing.mileage)} كم`],
          ["الحالة", ar(listing.condition)],
          ["المدينة", ar(listing.city)],
          ["الفئة", listing.trim || "—"],
        ].map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
    </Dialog>
  );
}
function Budget({ meta }: { meta: DatasetMeta | null }) {
  const [filter, setFilter] = useState<Filters>({}),
    [results, setResults] = useState<Listing[] | null>(null),
    [detail, setDetail] = useState<Listing | null>(null),
    [message, setMessage] = useState(""),
    [seenIds, setSeenIds] = useState<string[]>([]),
    [seenModels, setSeenModels] = useState<string[]>([]),
    [loading, setLoading] = useState(false),
    [hasSearched, setHasSearched] = useState(false);
  const set = (key: keyof Filters) => (value: string) =>
    setFilter((f) => ({
      ...f,
      [key]: value,
      ...(key === "make" ? { model: "" } : {}),
    }));
  const find = async (fresh = true) => {
    if (filter.min && filter.max && Number(filter.min) > Number(filter.max)) { setMessage("الحد الأدنى يجب ألا يتجاوز الحد الأعلى للميزانية."); return; }
    setLoading(true);
    try {
      let ids = fresh ? [] : seenIds;
      let models = fresh ? [] : seenModels;
      let found = await uiService.recommendations(filter, ids, models);
      if (!fresh && !found.items.length && seenIds.length) {
        // بعد عرض كل الموديلات الممكنة، اسمح بموديل سبق عرضه لكن بإعلان مختلف.
        found = await uiService.recommendations(filter, seenIds, []);
        models = [];
      }
      if (!fresh && !found.items.length && seenIds.length) {
        // إذا استُهلكت الإعلانات المطابقة كلها، ابدأ دورة جديدة بدل بقاء الزر بلا نتيجة.
        ids = [];
        models = [];
        found = await uiService.recommendations(filter, [], []);
      }
      setSeenIds([...new Set([...(ids), ...found.items.map((v) => v.id)])]);
      setSeenModels([...new Set([...(models), ...found.items.map((v) => `${v.make}|${v.model}`)])]);
      setResults(found.items);
      setHasSearched(true);
      setMessage(found.items.length ? "" : "انتهت الإعلانات المطابقة دون تكرار الموديل. غيّر الفلاتر لعرض خيارات أخرى.");
    } catch (error) { setMessage(error instanceof Error ? error.message : "تعذر تحميل التوصيات."); }
    finally { setLoading(false); }
  };
  const shown = results;
  return (
    <Frame page="budget">
      <Feedback text={message} close={()=>setMessage("")}/>
      <StaticArt
        page="budget"
        id="heading"
        text="اعثر على سيارتك المناسبة لميزانيتك — اختر ميزانيتك وتصفح أفضل 5 سيارات من الإعلانات المتاحة لدينا"
        heading
      />
      <GlassCard page="budget" id="filters" label="حدد ميزانيتك وتفضيلاتك">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void find();
          }}
        >
          <Field
            at={{ x: 1053, y: 106, w: 229, h: 39 }}
            label="الحد الأدنى"
            value={filter.min || ""}
            onChange={set("min")}
            type="number"
          />
          <Field
            at={{ x: 800, y: 106, w: 225, h: 39 }}
            label="الحد الأعلى"
            value={filter.max || ""}
            onChange={set("max")}
            type="number"
          />
          <Field
            at={{ x: 556, y: 106, w: 212, h: 39 }}
            label="اختر الشركة"
            value={filter.make || ""}
            onChange={set("make")}
            choices={meta?.makes || []}
          />
          <Field
            at={{ x: 305, y: 106, w: 221, h: 39 }}
            label="اختر الموديل"
            value={filter.model || ""}
            onChange={set("model")}
            choices={meta?.models_by_make?.[filter.make || ""] || []}
          />
          <Field
            at={{ x: 44, y: 106, w: 228, h: 39 }}
            label="اختر سنة الصنع"
            value={filter.year || ""}
            onChange={set("year")}
            choices={meta?.years.map(String) || []}
          />
          <Action
            at={{ x: 493, y: 164, w: 338, h: 49 }}
            type="submit"
            label="اعرض السيارات"
          />
        </form>
      </GlassCard>
      <StaticArt
        page="budget"
        id="results-heading"
        text="أفضل 5 سيارات حسب ميزانيتك"
      />
      <StaticArt
        page="budget"
        id="notice"
        text="الإعلانات المعروضة من قاعدة بيانات المشروع وقد لا تكون متاحة حالياً"
      />
      {[0, 1, 2, 3, 4].map((i) => (
        <GlassCard
          key={i}
          page="budget"
          id={`car${i}`}
          label="الشركة / الموديل — السعر من الإعلان"
        >
          {results && (
            <div className="result-car">
              {shown?.[i] ? (
                <>
                  {shown[i].image && <img
                    src={shown[i].image}
                    alt={`${ar(shown[i].make)} ${shown[i].model}`}
                    loading="lazy"
                    decoding="async"
                  />}
                  <b>{vehicleName(shown[i].make, shown[i].model)}</b>
                  <span>{number(shown[i].price)} ريال</span>
                  <small className="budget-card-meta">{shown[i].year ?? "السنة غير متوفرة"} · {shown[i].mileage == null ? "الممشى غير متوفر" : `${number(shown[i].mileage)} كم`} · {number(shown[i].imageCount)} صور</small>
                </>
              ) : (
                <span className="empty-result">—</span>
              )}
            </div>
          )}
          <Action
            at={{ x: 12, y: 249, w: part("budget", `car${i}`).w - 26, h: 43 }}
            label={`عرض التفاصيل ${i + 1}`}
            onClick={() => { if (shown?.[i]) setDetail(shown[i]); }}
            disabled={loading}
          />
        </GlassCard>
      ))}
      <PrimaryButton
        page="budget"
        id="refresh"
        label="غيّرها"
        onClick={() => void find(!hasSearched)}
      />
      {loading && <div className="identity-feedback" role="status">جارٍ البحث في الإعلانات…</div>}
      {detail && (
        <ListingDetails listing={detail} onClose={() => setDetail(null)} />
      )}
    </Frame>
  );
}
function Picker({
  excludeId,
  onSelect,
  onClose,
}: {
  excludeId?: string;
  onSelect: (v: Listing) => void;
  onClose: () => void;
}) {
  const [q, setQ] = useState(""), [items, setItems] = useState<Listing[]>([]), [offset, setOffset] = useState(0), [total, setTotal] = useState(0), [loading, setLoading] = useState(false);
  const requestId = useRef(0);
  const [error, setError] = useState("");
  const load = async (nextOffset: number, query: string) => {
    const id = ++requestId.current;
    setLoading(true); setError("");
    try {
      const page = await uiService.listingsPage({query}, nextOffset, 30);
      if (id !== requestId.current) return;
      setItems(old => nextOffset ? [...old, ...page.items] : page.items);
      setTotal(page.total); setOffset(nextOffset);
    } catch {
      if (id === requestId.current) setError("تعذر تحميل الإعلانات. حاول مرة أخرى.");
    } finally { if (id === requestId.current) setLoading(false); }
  };
  useEffect(() => {
    const timer = window.setTimeout(() => void load(0, q), 250);
    return () => { window.clearTimeout(timer); requestId.current++; };
  }, [q]);
  return (
    <Dialog label="اختر إعلانًا" onClose={onClose}>
      <input
        className="picker-search"
        aria-label="الشركة / الموديل"
        placeholder="الشركة / الموديل"
        value={q}
        onChange={(e) => setQ(e.target.value)}
      />
      {error && <p role="alert">{error}</p>}
      <p className="history-note">{number(total)} إعلان مسجل في قاعدة المشروع</p>
      <div className="picker-list">
        {items.filter(v => v.id !== excludeId)
          .map((v) => (
            <button key={v.id} onClick={() => onSelect(v)}>
              {v.image && <img src={v.image} alt="" />}
              <span>
                <b>
                  {vehicleName(v.make, v.model)}
                </b>
                <small>
                  {v.year} · {number(v.mileage)} كم
                </small>
              </span>
              <strong>{number(v.price)} ريال</strong>
            </button>
          ))}
      </div>
      {loading && <p>جارٍ تحميل الإعلانات…</p>}
      {!loading && offset + 30 < total && <button className="live-primary" onClick={() => void load(offset + 30, q)}>تحميل المزيد من الإعلانات</button>}
    </Dialog>
  );
}
function Compare() {
  const [cars, setCars] = useState<(Listing | null)[]>([null, null]),
    [pick, setPick] = useState<number | null>(null),
    [compare, setCompare] = useState(false);
  const rows = (v: Listing) => [
    vehicleName(v.make, v.model),
    String(v.year ?? '—'),
    `${number(v.price)} ريال`,
    `${number(v.mileage)} كم`,
    ar(v.condition),
    ar(v.city),
    [v.trim, ar(v.transmission)].filter(Boolean).join(" · "),
  ];
  return (
    <Frame page="compare">
      <StaticArt
        page="compare"
        id="heading"
        text="قارن بين سيارتين — اختر إعلانين لمقارنة المواصفات والميزات بسهولة"
        heading
      />
      {["first", "second"].map((id, i) => (
        <GlassCard
          key={id}
          page="compare"
          id={id}
          label={i ? "السيارة الثانية" : "السيارة الأولى"}
        >
          {cars[i] ? (
            <div className="selected-comparison">
              {cars[i]!.image ? (
                <img src={cars[i]!.image} alt={vehicleName(cars[i]!.make, cars[i]!.model)} />
              ) : (
                <div className="selected-comparison-no-image">{vehicleName(cars[i]!.make, cars[i]!.model)}</div>
              )}
              <div className="selected-comparison-caption">
                <b>{vehicleName(cars[i]!.make, cars[i]!.model)}</b>
                <span>{cars[i]!.year ?? "السنة غير متوفرة"} · {number(cars[i]!.price)} ريال</span>
              </div>
              <button
                className="comparison-remove"
                type="button"
                aria-label={`إزالة السيارة ${i + 1} المختارة`}
                title="إزالة السيارة المختارة"
                onClick={() => {
                  setCars((current) => current.map((car, index) => index === i ? null : car));
                  setCompare(false);
                }}
              >
                ×
              </button>
            </div>
          ) : (
            <Action
              at={{ x: 315, y: 75, w: 276, h: 66 }}
              label={i ? "اختر الإعلان الثاني" : "اختر الإعلان الأول"}
              onClick={() => setPick(i)}
            />
          )}
        </GlassCard>
      ))}
      <GlassCard page="compare" id="table" label="جدول مقارنة السيارات">
        <table className={compare ? "comparison-values" : "sr-only"}>
          <thead>
            <tr>
              <th>عنصر المقارنة</th>
              <th>السيارة الأولى</th>
              <th>السيارة الثانية</th>
            </tr>
          </thead>
          <tbody>
            {[
              "الماركة والموديل",
              "سنة الصنع",
              "السعر المعلن",
              "الممشى",
              "الحالة",
              "المدينة",
              "المواصفات المتاحة",
            ].map((label, index) => (
              <tr key={label}>
                <th>{label}</th>
                {cars.map((v, i) => (
                  <td key={i}>{v && compare ? rows(v)[index] : "—"}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </GlassCard>
      <PrimaryButton
        page="compare"
        id="submit"
        label="ابدأ المقارنة"
        onClick={() => {
          if (!cars[0]) setPick(0);
          else if (!cars[1]) setPick(1);
          else setCompare(true);
        }}
      />
      <GlassCard
        page="compare"
        id="information"
        label="يظهر ملخص الفروقات بعد اختيار الإعلانين"
      >
        {compare && cars[0] && cars[1] && (
            <div className="comparison-summary">
              {(() => {
                const first = cars[0]!, second = cars[1]!;
                const numeric = (a: number | null, b: number | null, label: string, unit: string, lowerIsBetter: boolean) => {
                  if (a == null || b == null) return {label, value:"غير متوفر في أحد الإعلانين", tone:"unknown"};
                  if (a === b) return {label, value:`متساويان · ${number(a)} ${unit}`, tone:"equal"};
                  const winner = (lowerIsBetter ? a < b : a > b) ? "الأولى" : "الثانية";
                  return {label, value:`السيارة ${winner} · الفرق ${number(Math.abs(a-b))} ${unit}`, tone:"value"};
                };
                const price = numeric(first.price, second.price, "السعر الأقل", "ريال", true);
                const mileage = numeric(first.mileage, second.mileage, "الممشى الأقل", "كم", true);
                const year = numeric(first.year, second.year, "سنة الصنع الأحدث", "", false);
                const details = [price,mileage,year];
                return <>
                  {details.map(item=><div className={`comparison-fact ${item.tone}`} key={item.label}><span>{item.label}</span><strong>{item.value}</strong></div>)}
                </>;
              })()}
            </div>
        )}
      </GlassCard>
      {pick !== null && (
        <Picker
          excludeId={cars[1 - pick]?.id}
          onClose={() => setPick(null)}
          onSelect={(v) => {
            setCars((c) => c.map((old, i) => (i === pick ? v : old)));
            setCompare(false);
            setPick(null);
          }}
        />
      )}
    </Frame>
  );
}
function ListingCards({ items, onSelect }: { items: Listing[]; onSelect: (v: Listing) => void }) {
  return <div className="browse-cards">{items.map((v) => <article className="live-glass browse-car" key={v.id}>
    {v.image && <img src={v.image} alt={vehicleName(v.make, v.model)} />}
    <div className="browse-car-tags"><span>{ar(v.condition) || "الحالة غير متوفرة"}</span><span>{number(v.imageCount)} صور</span></div>
    <h2>{vehicleName(v.make, v.model)}</h2><p>{v.year ?? "—"} · {number(v.mileage)} كم · {ar(v.city) || "المدينة غير متوفرة"}</p>
    <strong>{number(v.price)} ريال</strong><button className="live-primary" onClick={() => onSelect(v)}>عرض التفاصيل</button>
  </article>)}</div>;
}
function MarketBar({label, count, total, detail, onClick}: {label: string; count: number; total: number; detail?: string; onClick?: () => void}) {
  const share = total ? 100 * count / total : 0;
  return <button className={`market-bar market-data-row${onClick ? " market-clickable" : ""}`} onClick={onClick} disabled={!onClick}>
    <span>{label}</span><div><i style={{width:`${Math.max(count ? 1.5 : 0, Math.min(100, share))}%`}}/></div>
    <b>{number(count)}</b><small>{share.toFixed(1)}%{detail ? ` · ${detail}` : " من العينة"}</small>
  </button>;
}
function MarketBarChart({items, total, onSelect}: {items:{key:string;label:string;count:number;median_price:number|null}[];total:number;onSelect?:(key:string)=>void}) {
  return <div className="market-bar-chart">{items.map(item=>{
    const share=total?100*item.count/total:0;
    return <button key={item.key} type="button" className={`market-chart-row${onSelect?" clickable":""}`} onClick={onSelect?()=>onSelect(item.key):undefined} disabled={!onSelect}>
      <span>{item.label}</span><div className="market-chart-track"><i style={{width:`${Math.max(item.count?1.5:0,share)}%`}}/></div><b>{number(item.count)}</b><small>{share.toFixed(1)}% من العينة{item.median_price==null?"":` · وسيط ${number(Math.round(item.median_price))} ريال`}</small>
    </button>;
  })}</div>;
}
function PriceBandChart({bands, total}: {bands: MarketData["price_bands"]; total: number}) {
  const colors = ["#00d7f5", "#138cff", "#635bff", "#a45cff"];
  let cursor = 0;
  const segments = bands.map((band, i) => {
    const start = cursor;
    cursor += total ? 360 * band.count / total : 0;
    return `${colors[i % colors.length]} ${start.toFixed(2)}deg ${cursor.toFixed(2)}deg`;
  });
  return <div className="price-band-chart"><div className="price-donut" style={{background:`conic-gradient(${segments.join(",")})`}}><div><strong>{number(total)}</strong><span>إعلان</span></div></div><div className="price-band-legend">{bands.map((band,i)=><div key={band.label}><i style={{background:colors[i%colors.length]}}/><span>{band.label}</span><b>{number(band.count)}</b><small>{total ? `${(100*band.count/total).toFixed(1)}%` : "0.0%"}</small></div>)}</div></div>;
}
function MedianBars({items, label, onSelect}: {items: {label:string; median_price:number|null; count:number; key:string}[]; label:string; onSelect?:(key:string)=>void}) {
  const max = Math.max(1, ...items.map(item=>item.median_price || 0));
  return <div className="median-bars" aria-label={label}>{items.map(item=><button key={item.key} type="button" className="median-bar-row" onClick={()=>onSelect?.(item.key)} disabled={!onSelect}><span>{item.label}</span><div><i style={{width:`${item.median_price==null?0:100*item.median_price/max}%`}}/></div><b>{item.median_price==null?"—":`${number(Math.round(item.median_price))} ريال`}</b><small>{number(item.count)} إعلان</small></button>)}</div>;
}
function VehicleYearChart({items}: {items: MarketData["by_vehicle_year"]}) {
  const valid = items.filter(item=>Number.isFinite(item.median_price));
  if (!valid.length) return <p className="history-note">لا توجد بيانات كافية للرسم.</p>;
  const min = Math.min(...valid.map(item=>item.median_price)), max = Math.max(...valid.map(item=>item.median_price));
  const points = valid.map((item,index)=>({ ...item, x: valid.length===1?500:40+920*index/(valid.length-1), y: 235-(max===min?100:190*(item.median_price-min)/(max-min)) }));
  return <div className="vehicle-year-chart"><svg viewBox="0 0 1000 280" role="img" aria-label="وسيط أسعار الإعلانات حسب سنة صنع السيارة"><defs><linearGradient id="year-line" x1="0" x2="1"><stop stopColor="#147cff"/><stop offset="1" stopColor="#00def2"/></linearGradient><linearGradient id="year-area" x1="0" x2="0" y1="0" y2="1"><stop stopColor="#00c9ed" stopOpacity=".28"/><stop offset="1" stopColor="#00c9ed" stopOpacity="0"/></linearGradient></defs>{[45,105,165,225].map(y=><line key={y} x1="30" x2="980" y1={y} y2={y} stroke="#245174" strokeDasharray="5 8"/>)}<polygon points={`30,235 ${points.map(point=>`${point.x},${point.y}`).join(" ")} 980,235`} fill="url(#year-area)"/><polyline points={points.map(point=>`${point.x},${point.y}`).join(" ")} fill="none" stroke="url(#year-line)" strokeWidth="7" strokeLinejoin="round" strokeLinecap="round"/>{points.map(point=><g key={point.year}><circle cx={point.x} cy={point.y} r="9" fill="#071c39" stroke="#33e3ff" strokeWidth="5"><title>{`${point.year}: وسيط ${number(point.median_price)} ريال، ${number(point.count)} إعلان`}</title></circle><text x={point.x} y="267" textAnchor="middle" fill="#c6def4" fontSize="24">{point.year}</text></g>)}</svg><div className="year-chart-caption"><span>وسيط السعر المعلن</span><strong>{number(Math.round(min))} – {number(Math.round(max))} ريال</strong><small>حسب سنة صنع السيارة؛ لا يمثل تغير السعر بمرور الزمن</small></div></div>;
}
function Supplemental({ page, meta }: { page: "market" | "browse"; meta: DatasetMeta | null }) {
  const [filter, setFilter] = useState<Filters>({ make: new URLSearchParams(location.search).get("make") || "" }),
    [detail, setDetail] = useState<Listing | null>(null),
    [items, setItems] = useState<Listing[]>([]), [total, setTotal] = useState(0),
    [offset, setOffset] = useState(0), [market, setMarket] = useState<MarketData | null>(null),
    [marketTab, setMarketTab] = useState("overview"),
    [loading, setLoading] = useState(false), [error, setError] = useState("");
  const requestId = useRef(0);
  const loadListings = async (next = 0) => {
    const id = ++requestId.current;
    setLoading(true); setError("");
    try { const result = await uiService.listingsPage(filter, next, 24); if (id !== requestId.current) return; setItems(result.items); setTotal(result.total); setOffset(next); }
    catch (e) { if (id === requestId.current) setError(e instanceof Error ? e.message : "تعذر تحميل الإعلانات."); }
    finally { if (id === requestId.current) setLoading(false); }
  };
  const loadMarket = async (selected: Filters = filter) => {
    const id = ++requestId.current;
    setLoading(true); setError("");
    try { const result = await uiService.market({make:selected.make, model:selected.model, year:selected.year, city:selected.city, min:selected.min, max:selected.max}); if (id === requestId.current) setMarket(result); }
    catch (e) { if (id === requestId.current) setError(e instanceof Error ? e.message : "تعذر تحميل تحليل السوق."); }
    finally { if (id === requestId.current) setLoading(false); }
  };
  useEffect(() => { if (page === "browse") void loadListings(0); else void loadMarket(); return () => { requestId.current++; }; }, [page]);
  const set = (key: keyof Filters) => (value: string) => setFilter((f) => ({...f,[key]:value,...(key === "make" ? {model:""} : {})}));
  const filterFields = <>
    <label>الشركة<select value={filter.make || ""} onChange={(e)=>set("make")(e.target.value)}><option value="">كل الشركات</option>{(meta?.makes||[]).map(m=><option key={m} value={m}>{ar(m)}</option>)}</select></label>
    <label>الموديل<select value={filter.model || ""} onChange={(e)=>set("model")(e.target.value)}><option value="">كل الموديلات</option>{(meta?.models_by_make?.[filter.make||""]||[]).map(m=><option key={m} value={m}>{m}</option>)}</select></label>
    <label>سنة الصنع<select value={filter.year || ""} onChange={(e)=>set("year")(e.target.value)}><option value="">كل السنوات</option>{(meta?.years||[]).map(y=><option key={y}>{y}</option>)}</select></label>
    <label>المدينة<select value={filter.city || ""} onChange={(e)=>set("city")(e.target.value)}><option value="">كل المدن</option>{(meta?.cities||[]).map(c=><option key={c}>{ar(c)}</option>)}</select></label>
  </>;
  const priceFields = <>
    <label>أقل سعر<input inputMode="numeric" type="number" min="0" placeholder="من ريال" value={filter.min||""} onChange={(e)=>set("min")(e.target.value)}/></label>
    <label>أعلى سعر<input inputMode="numeric" type="number" min="0" placeholder="إلى ريال" value={filter.max||""} onChange={(e)=>set("max")(e.target.value)}/></label>
  </>;
  const priceShare = (count:number) => market?.count ? `${(100*count/market.count).toFixed(1)}%` : "0.0%";
  const largestBand = market?.price_bands.slice().sort((a,b)=>b.count-a.count)[0];
  const under70k = market?.price_bands.filter(b=>b.max!==null&&b.max<=70000).reduce((sum,b)=>sum+b.count,0) ?? 0;
  const meanMedianGap = market?.median_price && market.average_price != null ? 100*(market.average_price-market.median_price)/market.median_price : null;
  const dateCoverage = market?.count ? 100*market.count_with_listing_date/market.count : 0;
  const topModel = market?.by_model[0];
  const marketTabs = [{id:"overview",label:"نظرة عامة"},{id:"brands",label:"الشركات والموديلات"},{id:"prices",label:"الأسعار والممشى"},{id:"years",label:"سنوات الصنع والمدن"}];
  return <div className="supplemental" dir="rtl">
    <div className="supplemental-nav"><Navbar page={page} /></div>
    <header className="supplemental-heading"><h1>{titles[page]}</h1><p>{page === "market" ? "استكشف مؤشرات الأسعار والإعلانات حسب الشركة والموديل والسنة والمدينة" : "تصفح إعلانات السيارات المحفوظة في قاعدة بيانات المشروع"}</p></header>
    <div className="supplemental-content">
      {page === "market" ? <>
        <section className="market-service-links" aria-label="خدمات تقدير السيارات">
          <a href="/sell" onClick={event => { event.preventDefault(); go("/sell"); }}><strong>بكم أبيع سيارتي؟</strong><span>أدخل بيانات السيارة للحصول على تقدير سعري</span><b>ابدأ التقدير ←</b></a>
          <a href="/estimate-from-image" onClick={event => { event.preventDefault(); go("/estimate-from-image"); }}><strong>تقدير سعر السيارة من الصورة</strong><span>ارفع صورة السيارة وانتقل لخدمة التقدير</span><b>قدّر من الصورة ←</b></a>
        </section>
        <section className="live-glass browse-filters market-filters">{filterFields}{priceFields}<button className="live-primary" onClick={()=>void loadMarket()}>تحديث التحليل</button></section>
        {error && <p className="history-note">{error}</p>}{loading && <p className="history-note">جارٍ تحديث المؤشرات…</p>}
        {market && <>
          <div className="market-metrics">{[
            ["إعلانات بسعر صالح",number(market.count),"العينة الحالية بعد الفلاتر"],
            ["وسيط السعر",market.median_price == null ? "—" : `${number(market.median_price)} ريال`,"القيمة الوسطى للإعلانات"],
            ["متوسط السعر",market.average_price == null ? "—" : `${number(Math.round(market.average_price))} ريال`,"يتأثر بالقيم المرتفعة"],
            ["نطاق الأسعار",market.min_price == null ? "—" : `${number(market.min_price)} – ${number(market.max_price)} ريال`,"أدنى وأعلى إعلان ضمن العينة"],
            ["وسيط الممشى",market.median_mileage == null ? "—" : `${number(market.median_mileage)} كم`,"للسجلات التي لديها ممشى"],
            ["تاريخ الإعلان متوفر",`${dateCoverage.toFixed(1)}%`,`${number(market.count_with_listing_date)} من ${number(market.count)} إعلان`],
          ].map(([label,value,note])=><section className="live-glass market-metric-card" key={label}><span>{label}</span><strong>{value}</strong><small>{note}</small></section>)}</div>
          <nav className="market-tabs" aria-label="أقسام تحليل السوق">{marketTabs.map(t=><button key={t.id} className={marketTab===t.id?"active":""} onClick={()=>setMarketTab(t.id)}>{t.label}</button>)}</nav>
          {marketTab === "overview" && <>
            <section className="market-insights"><div className="live-glass"><span>الشريحة السعرية الأكبر</span><strong>{largestBand?.label||"—"}</strong><small>{largestBand?`${number(largestBand.count)} إعلان · ${priceShare(largestBand.count)}`:"لا توجد بيانات"}</small></div><div className="live-glass"><span>الإعلانات تحت 70,000 ريال</span><strong>{priceShare(under70k)}</strong><small>{number(under70k)} إعلان ضمن الشرائح الأقل من 70 ألف</small></div><div className="live-glass"><span>فرق المتوسط عن الوسيط</span><strong>{meanMedianGap==null?"—":`${meanMedianGap>=0?"+":""}${meanMedianGap.toFixed(1)}%`}</strong><small>مؤشر وصفي لتباعد الأسعار، وليس نموًا زمنيًا</small></div><div className="live-glass"><span>أكثر موديل ظهورًا</span><strong>{topModel?`${vehicleName(topModel.make,topModel.model)}`:"—"}</strong><small>{topModel?`${number(topModel.count)} إعلان · ${priceShare(topModel.count)}`:"لا توجد بيانات"}</small></div></section>
            <div className="market-grid">
              <section className="live-glass market-chart"><h2>توزيع الأسعار في العينة</h2><PriceBandChart bands={market.price_bands} total={market.count}/></section>
              <section className="live-glass market-chart"><h2>الشركات الأكثر ظهورًا</h2><MarketBarChart items={market.by_make.slice(0,8).map(v=>({key:v.make,label:`${ar(v.make)} (${v.make})`,count:v.count,median_price:v.median_price}))} total={market.count} onSelect={key=>{const selected={...filter,make:key,model:""};setFilter(selected);void loadMarket(selected);}}/></section>
              <section className="live-glass market-chart"><h2>وسيط السعر في أكثر الموديلات ظهورًا</h2><MedianBars label="وسيط السعر حسب الموديل" items={market.by_model.slice(0,8).map(v=>({key:`${v.make}|${v.model}`,label:vehicleName(v.make,v.model),count:v.count,median_price:v.median_price}))} onSelect={key=>{const [make,model]=key.split("|");const selected={...filter,make,model};setFilter(selected);void loadMarket(selected);}}/></section>
            </div>
          </>}
          {marketTab === "brands" && <div className="market-grid"><section className="live-glass market-chart"><h2>حصة الإعلانات حسب الشركة</h2><MarketBarChart items={market.by_make.slice(0,18).map(v=>({key:v.make,label:`${ar(v.make)} (${v.make})`,count:v.count,median_price:v.median_price}))} total={market.count} onSelect={key=>{const selected={...filter,make:key,model:""};setFilter(selected);void loadMarket(selected);}}/></section><section className="live-glass market-chart"><h2>الموديلات الأكثر ظهورًا</h2><MarketBarChart items={market.by_model.map(v=>({key:`${v.make}|${v.model}`,label:vehicleName(v.make,v.model),count:v.count,median_price:v.median_price}))} total={market.count} onSelect={key=>{const [make,model]=key.split("|");const selected={...filter,make,model};setFilter(selected);void loadMarket(selected);}}/></section></div>}
          {marketTab === "prices" && <div className="market-grid"><section className="live-glass market-chart"><h2>وسيط سعر الإعلان حسب الممشى</h2><MedianBars label="وسيط السعر حسب شرائح الممشى" items={market.mileage_bands.map(v=>({key:v.label,label:v.label,count:v.count,median_price:v.median_price}))}/></section><section className="live-glass market-chart"><h2>توزيع الإعلانات حسب السعر</h2><PriceBandChart bands={market.price_bands} total={market.count}/></section></div>}
          {marketTab === "years" && <div className="market-grid"><section className="live-glass market-chart"><h2>وسيط الأسعار حسب سنة صنع السيارة</h2><VehicleYearChart items={market.by_vehicle_year}/></section><section className="live-glass market-chart"><h2>الإعلانات حسب المدينة</h2><MarketBarChart items={market.by_city.map(v=>({key:v.city,label:ar(v.city),count:v.count,median_price:v.median_price}))} total={market.count}/></section>{market.by_listing_year.length>0&&<section className="live-glass market-chart"><h2>سنة تسجيل الإعلان في البيانات</h2><MarketBarChart items={market.by_listing_year.map(v=>({key:String(v.year),label:String(v.year),count:v.count,median_price:v.median_price}))} total={market.count}/></section>}</div>}
          <p className="history-note market-data-note">{market.notice}</p>
        </>}
      </> : <>
        <section className="live-glass browse-filters browse-filter-panel"><label>ابحث في الإعلانات<input type="search" placeholder="ماركة، موديل، مدينة أو فئة" value={filter.query||""} onChange={(e)=>set("query")(e.target.value)}/></label>{filterFields}{priceFields}<label>ترتيب النتائج<select value={filter.sortBy||"newest"} onChange={(e)=>set("sortBy")(e.target.value)}><option value="newest">الأحدث تاريخًا في البيانات</option><option value="price_asc">السعر: من الأقل للأعلى</option><option value="price_desc">السعر: من الأعلى للأقل</option><option value="mileage_asc">الممشى: من الأقل للأعلى</option><option value="year_desc">سنة الصنع: الأحدث</option><option value="year_asc">سنة الصنع: الأقدم</option></select></label><button className="live-primary" onClick={()=>void loadListings(0)}>اعرض الإعلانات</button></section>
        {error&&<p className="history-note">{error}</p>}{loading&&<p className="history-note">جارٍ تحميل الإعلانات…</p>}
        <ListingCards items={items} onSelect={setDetail}/>
        <div className="browse-pagination"><button className="live-primary" disabled={offset===0||loading} onClick={()=>void loadListings(Math.max(0,offset-24))}>السابق</button><span>{number(Math.min(total,offset+1))}–{number(Math.min(total,offset+items.length))} من {number(total)}</span><button className="live-primary" disabled={offset+24>=total||loading} onClick={()=>void loadListings(offset+24)}>التالي</button></div>
      </>}
    </div>
    {detail&&<ListingDetails listing={detail} onClose={()=>setDetail(null)}/>}
  </div>;
}
export default function App() {
  const [page, setPage] = useState(guardedRoute),
    [sellPreview, setSellPreview] = useState(""),
    [sellFile, setSellFile] = useState<File | null>(null),
    [sellRevision, setSellRevision] = useState(0),
    [manual, setManual] = useState(false),
    [uploadError, setUploadError] = useState(""),
    [meta, setMeta] = useState<DatasetMeta | null>(null),
    [pricePairs, setPricePairs] = useState<Record<string, string[]>>({});
  const upload = useRef<HTMLInputElement>(null);
  useEffect(() => {
    const changed = () => setPage(guardedRoute());
    window.addEventListener("popstate", changed);
    return () => window.removeEventListener("popstate", changed);
  }, []);
  useEffect(() => {
    document.title = `${titles[page]} | ${APP_NAME}`;
  }, [page]);
  useEffect(() => {
    let active = true;
    void Promise.all([uiService.meta(), uiService.pricingOptions()]).then(([dataset, prices]) => {
      if (!active) return;
      setMeta(dataset);
      setPricePairs(prices.pairs || {});
    }).catch((error: unknown) => {
      if (active) setUploadError(error instanceof Error ? error.message : "تعذر تحميل بيانات السيارات.");
    });
    return () => { active = false; };
  }, []);
  useEffect(
    () => () => {
      if (sellPreview) URL.revokeObjectURL(sellPreview);
    },
    [sellPreview],
  );
  return (
    <>
      <input
        ref={upload}
        className="file-input"
        aria-label="رفع صورة سيارتك"
        type="file"
        accept="image/*"
        onChange={async (e) => {
          const f = e.target.files?.[0];
          if (!f) return;
          try { setSellPreview(await prepareImage(f)); setSellFile(f); setSellRevision(revision=>revision+1); setManual(false); go("/sell/details"); } catch(error) {setUploadError((error as Error).message);}
          if(upload.current) upload.current.value = "";
        }}
      />
      <Feedback text={uploadError} close={()=>setUploadError("")}/>
      {page === "home" && <Home />}
      {page === "sell" && (
        <Sell
          onUpload={() => upload.current?.click()}
          onManual={() => {
            setManual(true);
            setSellPreview("");
            setSellFile(null);
            go("/sell/details");
          }}
        />
      )}
      {page === "details" && <Details key={`${sellRevision}-${sellFile ? `${sellFile.name}-${sellFile.lastModified}-${sellFile.size}` : manual ? "manual" : "empty"}`} preview={sellPreview} manual={manual} imageFile={sellFile} meta={meta} pricePairs={pricePairs} onReplacePhoto={() => upload.current?.click()} />}
      {page === "estimate" && <Estimate />}
      {page === "budget" && <Budget meta={meta} />}
      {page === "compare" && <Compare />}
      {page === "login" && <LoginPage navigate={go} header={null} />}
      {(page === "market" || page === "browse") && <Supplemental page={page} meta={meta} />}
    </>
  );
}


