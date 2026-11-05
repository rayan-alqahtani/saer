import { validateImage } from "./imageUpload";

export type Listing = {
  id: string;
  make: string;
  model: string;
  year: number | null;
  price: number | null;
  mileage: number | null;
  city: string;
  condition: string;
  trim: string;
  transmission: string;
  fuel: string;
  body: string;
  color: string;
  image: string;
  imageCount: number;
  historical: boolean;
};
export type Filters = {
  min?: string;
  max?: string;
  make?: string;
  model?: string;
  year?: string;
  city?: string;
  query?: string;
  sortBy?: string;
};
export type DatasetMeta = {
  app_name: string;
  listings_count: number;
  images_count: number;
  makes: string[];
  years: number[];
  cities: string[];
  trims: string[];
  models_by_make: Record<string, string[]>;
  notice: string;
};
export type MarketFilters = { make?: string; model?: string; year?: string; city?: string; min?: string; max?: string };
export type MarketData = {
  count: number;
  count_with_listing_date: number;
  median_price: number | null;
  average_price: number | null;
  min_price: number | null;
  max_price: number | null;
  median_mileage: number | null;
  price_bands: { label: string; min: number; max: number | null; count: number }[];
  mileage_bands: { label: string; count: number; median_price: number | null }[];
  by_make: { make: string; count: number; median_price: number | null }[];
  by_model: { make: string; model: string; count: number; median_price: number | null }[];
  by_city: { city: string; count: number; median_price: number | null }[];
  by_vehicle_year: { year: number; count: number; median_price: number }[];
  by_listing_year: { year: number; count: number; median_price: number }[];
  year_range: [number, number] | null;
  notice: string;
};

function mapListing(row: any): Listing {
  return {
    id: row.listing_id,
    make: row.make || "",
    model: row.model || "",
    year: row.year ?? null,
    price: row.price_sar ?? null,
    mileage: row.mileage_km ?? null,
    city: row.city || "",
    condition: row.condition || "",
    trim: row.trim || "",
    transmission: row.transmission || "",
    fuel: row.fuel || "",
    body: row.body || "",
    color: row.color || "",
    image: row.image_url || "",
    imageCount: row.image_count || 0,
    historical: row.historical !== false,
  };
}

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  const result = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(result.detail || "تعذر تحميل بيانات المشروع.");
  return result as T;
}
async function postJson<T>(url: string, payload: unknown): Promise<T> {
  const response = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  const result = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(result.detail || "تعذر إكمال التقدير. تحقق من البيانات وحاول مجددًا.");
  return result as T;
}
async function postImage<T>(url: string, file: File): Promise<T> {
  const body = new FormData();
  body.append("file", file);
  const response = await fetch(url, { method: "POST", body });
  const result = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(result.detail || "تعذر تحليل الصورة. جرّب صورة أخرى.");
  return result as T;
}

function listingParams(filters: Filters = {}, offset = 0, limit = 24) {
  const params = new URLSearchParams({ offset: String(offset), limit: String(limit) });
  if (filters.query) params.set("q", filters.query);
  if (filters.make) params.set("make", filters.make);
  if (filters.model) params.set("model", filters.model);
  if (filters.year) params.set("year", filters.year);
  if (filters.min) params.set("min_price", filters.min);
  if (filters.max) params.set("max_price", filters.max);
  if (filters.city) params.set("city", filters.city);
  if (filters.sortBy) params.set("sort_by", filters.sortBy);
  return params;
}

export const uiService = {
  async meta() {
    if (import.meta.env.VITE_DEMO_MODE === "true") return (await import("./demoService")).meta();
    return getJson<DatasetMeta>("/api/meta");
  },
  async pricingOptions() {
    if (import.meta.env.VITE_DEMO_MODE === "true") return (await import("./demoService")).pricingOptions();
    return getJson<{ pairs: Record<string, string[]>; notice: string }>("/api/pricing-options");
  },
  async vehicleOptions(make: string, model: string, year?: string) {
    if (import.meta.env.VITE_DEMO_MODE === "true") return (await import("./demoService")).vehicleOptions(make, model, year);
    const collect = async (filters: Filters) => {
      const first = await uiService.listingsPage(filters,0,100);
      const rows = [...first.items];
      for (let offset=100; offset<first.total; offset+=100) rows.push(...(await uiService.listingsPage(filters,offset,100)).items);
      return rows;
    };
    const rows = await collect({make,model});
    const years = [...new Set(rows.map(item=>item.year).filter((value):value is number=>value!=null))].sort((a,b)=>b-a);
    const trims = [...new Set(rows.map(item=>item.trim).filter(Boolean))].sort((a,b)=>a.localeCompare(b));
    const trimsByYear: Record<string,string[]> = {};
    for (const row of rows) if (row.year != null && row.trim) (trimsByYear[String(row.year)] ||= []).push(row.trim);
    for (const [key, values] of Object.entries(trimsByYear)) trimsByYear[key] = [...new Set(values)].sort((a,b)=>a.localeCompare(b));
    return {years,trims,trimsByYear};
  },
  async listingsPage(filters: Filters = {}, offset = 0, limit = 24) {
    if (import.meta.env.VITE_DEMO_MODE === "true") return (await import("./demoService")).listingsPage(filters, offset, limit);
    const result = await getJson<{ items: any[]; total: number; offset: number; limit: number }>(`/api/listings?${listingParams(filters, offset, limit)}`);
    return { ...result, items: result.items.map(mapListing) };
  },
  async listingImages(id: string) {
    if (import.meta.env.VITE_DEMO_MODE === "true") return (await import("./demoService")).listingImages(id);
    return getJson<{ images: { image_uid: string; url: string }[] }>(`/api/listings/${encodeURIComponent(id)}`);
  },
  async recommendations(filters: Filters, excludeIds: string[] = [], excludeModels: string[] = []) {
    if (import.meta.env.VITE_DEMO_MODE === "true") return (await import("./demoService")).recommendations(filters, excludeIds, excludeModels);
    const params = new URLSearchParams({ min_price: filters.min || "0", max_price: filters.max || "1000000000" });
    if (filters.make) params.set("make", filters.make);
    if (filters.model) params.set("model", filters.model);
    if (filters.year) params.set("year", filters.year);
    if (excludeIds.length) params.set("exclude", excludeIds.join(","));
    if (excludeModels.length) params.set("exclude_models", excludeModels.join(","));
    const result = await getJson<{ items: any[]; count: number; eligible_listing_count: number; eligible_model_count: number; notice: string }>(`/api/recommendations?${params}`);
    return { ...result, items: result.items.map(mapListing) };
  },
  async market(filters: MarketFilters) {
    if (import.meta.env.VITE_DEMO_MODE === "true") return (await import("./demoService")).market(filters);
    const params = new URLSearchParams();
    if (filters.make) params.set("make", filters.make);
    if (filters.model) params.set("model", filters.model);
    if (filters.year) params.set("year", filters.year);
    if (filters.city) params.set("city", filters.city);
    if (filters.min) params.set("min_price", filters.min);
    if (filters.max) params.set("max_price", filters.max);
    return getJson<MarketData>(`/api/market?${params}`);
  },
  async estimateFromImage(file: File) {
    validateImage(file);
    if (import.meta.env.VITE_DEMO_MODE === "true") return (await import("./demoService")).estimateFromImage(file);
    const result = await postImage<{price_sar?:number; range_80_sar?:[number,number]; range_90_sar?:[number,number]; notice_ar?:string; notice?:string}>("/api/estimate-image-price", file);
    const price = result.price_sar;
    if (typeof price !== "number" || !Number.isFinite(price)) throw new Error("لم يُرجع نموذج الصورة سعرًا صالحًا.");
    return { price, lower80: result.range_80_sar?.[0], upper80: result.range_80_sar?.[1], lower90: result.range_90_sar?.[0], upper90: result.range_90_sar?.[1], notice: result.notice_ar || result.notice || "تقدير تجريبي من الصورة." };
  },
  async estimateFromImages(files: File[]) {
    if (files.length !== 1) throw new Error("ارفع صورة واحدة فقط للسيارة.");
    if (import.meta.env.VITE_DEMO_MODE === "true") return (await import("./demoService")).estimateFromImages(files);
    const body = new FormData();
    files.forEach((file) => body.append("files", file));
    const response = await fetch("/api/estimate-image-price/multiple", { method: "POST", body });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "تعذر تقدير السعر.");
    const price = result.price_sar;
    if (typeof price !== "number" || !Number.isFinite(price)) throw new Error("لم يُرجع نموذج الصورة سعرًا صالحًا.");
    return { price, lower80: result.range_80_sar?.[0], upper80: result.range_80_sar?.[1], lower90: result.range_90_sar?.[0], upper90: result.range_90_sar?.[1], imageCount: result.image_count ?? files.length, notice: result.notice_ar || result.notice || "تقدير تجريبي من الصور." };
  },
  async identifyVehicle(file: File) {
    validateImage(file);
    if (import.meta.env.VITE_DEMO_MODE === "true") return (await import("./demoService")).identifyVehicle(file);
    const body = new FormData();
    body.append("file", file);
    const response = await fetch("/api/recognize", { method: "POST", body });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "تعذر التعرف على السيارة.");
    const suggestions = result["الاقتراح_الرئيسي"]?.["أفضل_الاحتمالات"] ?? [];
    const parse = (item: { الفئة: string; درجة_غير_معايرة: number }) => {
      const [make, ...model] = item["الفئة"].split("|");
      return { make, model: model.join("|"), confidence: item["درجة_غير_معايرة"] };
    };
    const alternatives = suggestions.map(parse);
    const predicted = result["الاقتراح_الرئيسي"];
    if (!predicted?.["الماركة"] || !predicted?.["الموديل"]) throw new Error("لم يستطع النموذج اقتراح الماركة والموديل لهذه الصورة.");
    return { make: predicted["الماركة"], model: predicted["الموديل"], confidence: predicted["درجة_غير_معايرة"] ?? 0, alternatives };
  },
  async estimateSale(details: Record<string, string>) {
    if (import.meta.env.VITE_DEMO_MODE === "true") return (await import("./demoService")).estimateSale(details);
    const result = await postJson<{سعر_الإعلان_المقدر_ريال?:number; notice?:string}>("/api/estimate-tabular", {
      make: details.make, model_std: details.model, year: Number(details.year), km_mileage: Number(details.mileage),
      trim_std: details.trim || "", transmission: details.transmission || "",
      type_fuel: details.fuel || "", condition: details.condition || "used",
    });
    if (typeof result.سعر_الإعلان_المقدر_ريال !== "number" || !Number.isFinite(result.سعر_الإعلان_المقدر_ريال)) throw new Error("لم يُرجع نموذج التسعير نتيجة صالحة.");
    return { price: result.سعر_الإعلان_المقدر_ريال, notice: result.notice || "تقدير إعلاني تجريبي." };
  },
};

const names: Record<string, string> = {
  Toyota: "تويوتا", Hyundai: "هيونداي", Honda: "هوندا", Chevrolet: "شيفروليه", Nissan: "نيسان",
  Kia: "كيا", Ford: "فورد", Mazda: "مازدا", Mitsubishi: "ميتسوبيشي", Suzuki: "سوزوكي", Lexus: "لكزس",
  BMW: "بي إم دبليو", Audi: "أودي", Volkswagen: "فولكس واجن", Dammam: "الدمام", Riyadh: "الرياض",
  Jeddah: "جدة", "Al Qurayyat": "القريات", Madinah: "المدينة المنورة", Najran: "نجران", used: "مستعملة", new: "جديدة", Automatic: "أوتوماتيك", Manual: "يدوي",
  BAIC: "بايك", BYD: "بي واي دي", Cadillac: "كاديلاك", Changan: "شانجان", Chery: "شيري", Chrysler: "كرايسلر",
  Deepal: "ديبال", Dodge: "دودج", Dongfeng: "دونغ فينغ", GAC: "جي إيه سي", GMC: "جي إم سي", Geely: "جيلي",
  Genesis: "جينيسيس", "Great Wall": "جريت وول", Haval: "هافال", Hongqi: "هونغتشي", Jaecoo: "جايكو", Jeep: "جيب",
  Jetour: "جيتور", "Lynk & Co": "لينك آند كو", MG: "إم جي", MINI: "ميني", Mercedes: "مرسيدس", Omoda: "أومودا",
  Porsche: "بورشه", Soueast: "سو إيست", SsangYong: "سانج يونج",
};
export const ar = (s: string | null | undefined) => (s ? names[s] || s : "—");
const modelNames: Record<string,string> = {
  "Tahoe":"تاهو", "Suburban":"سوبربان", "Silverado":"سيلفرادو", "Camaro":"كامارو", "Cruze":"كروز", "Malibu":"ماليبو", "Yukon":"يوكن", "Captiva":"كابتيفا", "Spark":"سبارك", "Trailblazer":"تريل بليزر", "Traverse":"ترافيرس",
  "Camry":"كامري", "Corolla":"كورولا", "Yaris":"يارس", "Fortuner":"فورتشنر", "RAV4":"راف فور", "Land Cruiser":"لاند كروزر", "Prado":"برادو", "Hilux":"هايلوكس", "Avalon":"أفالون", "Raize":"رايز", "Rush":"راش", "Innova":"إنوفا", "Crown":"كراون", "Land Cruiser Prado":"لاند كروزر برادو",
  "Accent":"أكسنت", "Elantra":"إلنترا", "Sonata":"سوناتا", "Tucson":"توسان", "Santa Fe":"سانتافي", "Grand i10":"جراند i10", "Creta":"كريتا", "Kona":"كونا", "Palisade":"باليسيد", "Azera":"أزيرا",
  "Sunny":"صني", "Altima":"ألتيما", "Patrol":"باترول", "X-Trail":"إكس تريل", "Kicks":"كيكس", "Sentra":"سنترا", "Maxima":"ماكسيما", "Urvan":"أورفان",
  "Sportage":"سبورتاج", "Pegas":"بيجاس", "Cerato":"سيراتو", "K5":"K5", "Sorento":"سورينتو", "Seltos":"سيلتوس", "Carnival":"كارنفال", "Rio":"ريو", "Picanto":"بيكانتو",
  "Dzire":"دزاير", "Swift":"سويفت", "Vitara":"فيتارا", "Jimny":"جمني", "Tiggo 4 Pro":"تيجو 4 برو", "Tiggo 7 Pro":"تيجو 7 برو", "Tiggo 8 Pro":"تيجو 8 برو", "Tiggo":"تيجو", "Arrizo 5":"أريزو 5", "T-Roc":"تي روك",
  "F-150":"F-150", "Explorer":"إكسبلورر", "Edge":"إيدج", "Escape":"إسكيب", "Expedition":"إكسبيديشن", "Mustang":"موستانج", "Ranger":"رينجر", "Taurus":"تورس", "Territory":"تيريتوري", "Charger":"تشارجر", "Challenger":"تشالنجر", "Alsvin":"ألسفن", "Eado Plus":"إيدو بلس", "Hunter":"هنتر", "Jolion":"جوليان", "Emgrand":"إمجراند", "RX5":"آر إكس 5", "GS3":"جي إس 3", "CX-5":"CX-5", "CX-9":"CX-9", "Mazda 6":"مازدا 6", "City":"سيتي", "Civic":"سيفيك", "Accord":"أكورد", "CR-V":"CR-V", "HR-V":"HR-V", "3":"3",
};
export const arModel = (model: string) => modelNames[model] || model;
export const vehicleName = (make: string | null | undefined, model: string | null | undefined) => {
  if (!make && !model) return "—";
  const arabic = [ar(make), model ? arModel(model) : ""].filter(Boolean).join(" ");
  const english = [make, model].filter(Boolean).join(" ");
  return `${arabic} (${english})`;
};
export const number = (n: number | null | undefined) => n == null ? "—" : new Intl.NumberFormat("en-US").format(n);
