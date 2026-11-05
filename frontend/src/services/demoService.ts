import type { DatasetMeta, Filters, Listing, MarketData, MarketFilters } from "./uiService";
import { Client, handle_file } from "@gradio/client";
import { resolveApiBaseUrl } from "../config";
import { validateImage } from "./imageUpload";

type RawListing = Record<string, any> & {
  listing_id: string;
  make: string;
  model: string;
  year: number | null;
  price_sar: number | null;
  mileage_km: number | null;
  city: string;
  trim: string;
  image_count: number;
  image_url: string | null;
  images: string[];
  listing_date?: string | null;
};
type DemoListings = { items: RawListing[] };
let listingsCache: Promise<DemoListings> | undefined;
let connectedApiUrl = "";
let clientPromise: Promise<Client> | undefined;

async function readJson<T>(name: string): Promise<T> {
  try {
    const response = await fetch(`/demo/${name}.json`);
    if (!response.ok) throw new Error();
    return await response.json() as T;
  } catch {
    throw new Error("تعذر تحميل بيانات العرض التجريبي.");
  }
}
const allListings = () => listingsCache ??= readJson<DemoListings>("listings").catch(error => {
  listingsCache = undefined;
  throw error;
});
const normalize = (value: unknown) => String(value ?? "").trim().toLocaleLowerCase();

function matches(row: RawListing, filters: Filters | MarketFilters = {}) {
  const f = filters as Filters;
  if (f.make && row.make !== f.make) return false;
  if (f.model && row.model !== f.model) return false;
  if (f.year && String(row.year ?? "") !== String(f.year)) return false;
  if (f.city && row.city !== f.city) return false;
  if (f.min && (row.price_sar == null || row.price_sar < Number(f.min))) return false;
  if (f.max && (row.price_sar == null || row.price_sar > Number(f.max))) return false;
  if (f.query) {
    const haystack = [row.make, row.model, row.city, row.trim].map(normalize).join(" ");
    if (!haystack.includes(normalize(f.query))) return false;
  }
  return true;
}
function toListing(row: RawListing): Listing {
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
function median(values: number[]) {
  if (!values.length) return null;
  const ordered = [...values].sort((a, b) => a - b);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2 ? ordered[middle] : (ordered[middle - 1] + ordered[middle]) / 2;
}
function summary(rows: RawListing[]) {
  const prices = rows.map(row => row.price_sar).filter((price): price is number => typeof price === "number" && price > 0 && Number.isFinite(price));
  const withImages = rows.filter(row => row.image_count > 0).length;
  return {
    priced_count: prices.length,
    median_price: median(prices),
    average_price: prices.length ? prices.reduce((sum, price) => sum + price, 0) / prices.length : null,
    with_images: withImages,
    image_coverage_pct: rows.length ? 100 * withImages / rows.length : 0,
  };
}

export async function meta() {
  return readJson<DatasetMeta>("meta");
}
export async function pricingOptions() {
  return readJson<{ pairs: Record<string, string[]>; notice: string }>("pricing-options");
}
export async function vehicleOptions(make: string, model: string, year?: string) {
  const { items } = await allListings();
  const rows = items.filter(row => row.make === make && row.model === model);
  const years = [...new Set(rows.map(row => row.year).filter((value): value is number => value != null))].sort((a, b) => b - a);
  const trims = [...new Set(rows.map(row => row.trim).filter(Boolean))].sort((a, b) => a.localeCompare(b));
  const trimsByYear: Record<string, string[]> = {};
  for (const row of rows) if (row.year != null && row.trim) (trimsByYear[String(row.year)] ||= []).push(row.trim);
  for (const [key, values] of Object.entries(trimsByYear)) trimsByYear[key] = [...new Set(values)].sort((a, b) => a.localeCompare(b));
  if (year) return { years, trims, trimsByYear };
  return { years, trims, trimsByYear };
}
export async function listingsPage(filters: Filters = {}, offset = 0, limit = 24) {
  const { items } = await allListings();
  let rows = items.filter(row => matches(row, filters));
  const sort = filters.sortBy || "newest";
  const value = (row: RawListing) => sort === "price_asc" ? row.price_sar : sort === "price_desc" ? row.price_sar : sort === "mileage_asc" ? row.mileage_km : row.year;
  if (sort !== "newest") rows = [...rows].sort((a, b) => {
    if (sort === "year_asc") return Number(value(a) ?? Number.MAX_SAFE_INTEGER) - Number(value(b) ?? Number.MAX_SAFE_INTEGER);
    if (sort === "year_desc") return Number(value(b) ?? Number.MIN_SAFE_INTEGER) - Number(value(a) ?? Number.MIN_SAFE_INTEGER);
    const result = Number(value(a) ?? Number.MAX_SAFE_INTEGER) - Number(value(b) ?? Number.MAX_SAFE_INTEGER);
    return sort === "price_desc" ? -result : result;
  });
  return { items: rows.slice(offset, offset + limit).map(toListing), total: rows.length, offset, limit, summary: summary(rows) };
}
export async function listingImages(id: string) {
  const { items } = await allListings();
  const row = items.find(item => item.listing_id === id);
  if (!row) throw new Error("الإعلان غير موجود في بيانات العرض.");
  return { images: (row.images || []).map((url, index) => ({ image_uid: `${id}-photo-${index + 1}`, url })) };
}
export async function recommendations(filters: Filters, excludeIds: string[] = [], excludeModels: string[] = []): Promise<{ items: Listing[]; count: number; eligible_listing_count: number; eligible_model_count: number; notice: string }> {
  const [data, saved] = await Promise.all([allListings(), readJson<any>("recommendations")]);
  const pool = [...saved.items, ...data.items].filter((row, index, rows) => rows.findIndex(candidate => candidate.listing_id === row.listing_id) === index);
  const excludedIds = new Set(excludeIds);
  const excludedModels = new Set(excludeModels.map(value => value.trim()));
  const grouped = new Map<string, RawListing[]>();
  for (const row of pool) {
    if (!matches(row, filters) || excludedIds.has(row.listing_id) || excludedModels.has(`${row.make}|${row.model}`)) continue;
    const key = `${row.make}|${row.model}`;
    grouped.set(key, [...(grouped.get(key) || []), row]);
  }
  const candidates = [...grouped.values()].map(rows => rows[0]).slice(0, 5);
  return { notice: String(saved.notice || ""), items: candidates.map(toListing), count: candidates.length, eligible_listing_count: [...grouped.values()].reduce((sum, rows) => sum + rows.length, 0), eligible_model_count: grouped.size };
}

function marketFromSample(rows: RawListing[]): MarketData {
  const valid = rows.filter(row => typeof row.price_sar === "number" && Number.isFinite(row.price_sar) && row.price_sar > 0);
  const prices = valid.map(row => row.price_sar as number);
  const groups = <T,>(keyOf: (row: RawListing) => T | null | undefined) => {
    const map = new Map<T, RawListing[]>();
    for (const row of valid) {
      const key = keyOf(row);
      if (key != null && key !== "") map.set(key, [...(map.get(key) || []), row]);
    }
    return map;
  };
  const byMake = groups(row => row.make);
  const byCity = groups(row => row.city);
  const byModel = groups(row => row.model ? `${row.make}|${row.model}` : null);
  const byYear = groups(row => row.year);
  const byListingYear = groups(row => row.listing_date?.slice(0, 4) || null);
  const makeGroups = [...byMake].map(([make, values]) => ({ make: String(make), count: values.length, median_price: median(values.map(row => row.price_sar as number)) })).sort((a, b) => b.count - a.count || a.make.localeCompare(b.make));
  const cityGroups = [...byCity].map(([city, values]) => ({ city: String(city), count: values.length, median_price: median(values.map(row => row.price_sar as number)) })).sort((a, b) => b.count - a.count || a.city.localeCompare(b.city));
  const modelGroups = [...byModel].map(([key, values]) => { const [make, model] = String(key).split("|"); return { make, model, count: values.length, median_price: median(values.map(row => row.price_sar as number)) }; }).sort((a, b) => b.count - a.count || a.make.localeCompare(b.make) || a.model.localeCompare(b.model)).slice(0, 10);
  const yearGroups = (map: Map<any, RawListing[]>) => [...map].map(([year, values]) => ({ year: Number(year), count: values.length, median_price: median(values.map(row => row.price_sar as number)) as number })).sort((a, b) => a.year - b.year);
  const priceBands = [{ label: "أقل من 40 ألف", min: 0, max: 40000 }, { label: "40–70 ألف", min: 40000, max: 70000 }, { label: "70–120 ألف", min: 70000, max: 120000 }, { label: "120 ألف فأكثر", min: 120000, max: null }].map(band => ({ ...band, count: prices.filter(price => price >= band.min && (band.max == null || price < band.max)).length }));
  const mileageBands = [{ label: "أقل من 50 ألف كم", min: 0, max: 50000 }, { label: "50–100 ألف كم", min: 50000, max: 100000 }, { label: "100–150 ألف كم", min: 100000, max: 150000 }, { label: "150 ألف كم فأكثر", min: 150000, max: null }].map(band => {
    const values = valid.filter(row => row.mileage_km != null && row.mileage_km >= band.min && (band.max == null || row.mileage_km < band.max)).map(row => row.price_sar as number);
    return { label: band.label, count: values.length, median_price: median(values) };
  });
  const dated = valid.filter(row => /^\d{4}/.test(row.listing_date || ""));
  const mileages = valid.map(row => row.mileage_km).filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  const years = valid.map(row => row.year).filter((value): value is number => typeof value === "number");
  return {
    count: prices.length,
    count_with_listing_date: dated.length,
    median_price: median(prices),
    average_price: prices.length ? prices.reduce((sum, price) => sum + price, 0) / prices.length : null,
    min_price: prices.length ? Math.min(...prices) : null,
    max_price: prices.length ? Math.max(...prices) : null,
    median_mileage: median(mileages),
    price_bands: priceBands,
    mileage_bands: mileageBands,
    by_make: makeGroups,
    by_model: modelGroups,
    by_city: cityGroups,
    by_vehicle_year: yearGroups(byYear),
    by_listing_year: yearGroups(byListingYear),
    year_range: years.length ? [Math.min(...years), Math.max(...years)] : null,
    notice: "تحليل وصفي لإعلانات المشروع وأسعار الطلب المسجلة، وليس للسوق الحي أو أسعار البيع النهائية. التوزيعات تتأثر بتغطية البيانات وتنوع السيارات.",
  };
}
export async function market(filters: MarketFilters) {
  const [saved, data] = await Promise.all([readJson<MarketData>("market"), allListings()]);
  const hasFilters = Object.values(filters).some(value => Boolean(value));
  return hasFilters ? marketFromSample(data.items.filter(row => matches(row, filters))) : saved;
}

async function gradioClient(): Promise<Client> {
  const url = await resolveApiBaseUrl();
  if (!url) throw new Error("لم يُضبط رابط خدمة النماذج. أضف ?api= أو ?apiConfig= إلى رابط الصفحة ثم أعد المحاولة.");
  if (url !== connectedApiUrl) {
    connectedApiUrl = url;
    const hostname = new URL(url).hostname.toLowerCase();
    const isNgrok = hostname.endsWith(".ngrok-free.app")
      || hostname.endsWith(".ngrok-free.dev")
      || hostname.endsWith(".ngrok.app");
    const options = isNgrok ? { headers: { "ngrok-skip-browser-warning": "true" } } : undefined;
    clientPromise = Client.connect(url, options).catch((error: unknown) => {
      connectedApiUrl = "";
      clientPromise = undefined;
      const detail = error instanceof Error ? error.message : String(error);
      throw new Error(`تعذر الاتصال بخدمة النماذج. تأكد من تشغيل جلسة Kaggle وصلاحية رابط الخادم، ثم أعد المحاولة. (${detail})`);
    });
  }
  return clientPromise!;
}

async function callLive<T>(endpoint: string, args: unknown[]): Promise<T> {
  try {
    const client = await gradioClient();
    const response = await client.predict(`/${endpoint}`, args);
    const data = (response as { data: unknown[] }).data;
    return data[0] as T;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (/fetch|network|timeout|503|sleep|wake|connect/i.test(message)) {
      throw new Error("الخادم قد يكون في بدء التشغيل أو عاد من وضع السكون. انتظر قليلاً ثم أعد المحاولة.");
    }
    throw new Error(message || "تعذر الاتصال بخدمة النماذج.");
  }
}

function priceResult(result: Record<string, any>) {
  const price = result.price_sar;
  if (typeof price !== "number" || !Number.isFinite(price)) throw new Error("لم يُرجع نموذج الصورة سعراً صالحاً.");
  const range80 = result.range_80_sar;
  const range90 = result.range_90_sar;
  return {
    price,
    lower80: range80?.[0], upper80: range80?.[1],
    lower90: range90?.[0], upper90: range90?.[1],
    imageCount: result.image_count ?? result.images_used ?? 1,
    notice: result.notice_ar || result.notice || "تقدير تجريبي من الصورة؛ ليس سعر بيع مضموناً.",
  };
}

export async function estimateFromImage(file: File) {
  validateImage(file);
  const client = await gradioClient();
  const config = client.config;
  const dependency = config?.dependencies.find(item => item.api_name === "predict_image");
  const component = config?.components.find(item => item.id === dependency?.inputs[0]);
  const count = component?.props.file_count;
  if (component?.type !== "file" || !["single", "multiple"].includes(String(count))) {
    throw new Error("إعداد مدخل الصورة في خدمة النماذج غير متوافق. تحقق من إصدار الخادم.");
  }
  const fileData = handle_file(file);
  // Outer array: positional arguments. Inner array: the ListFiles parameter only.
  return priceResult(await callLive<Record<string, any>>("predict_image", [count === "multiple" ? [fileData] : fileData]));
}
export async function estimateFromImages(files: File[]) {
  if (files.length !== 1) throw new Error("ارفع صورة واحدة فقط للسيارة.");
  return estimateFromImage(files[0]);
}
export async function identifyVehicle(file: File) {
  validateImage(file);
  const result = await callLive<Record<string, any>>("identify_vehicle", [handle_file(file)]);
  const predicted = result["الاقتراح_الرئيسي"];
  const alternatives = predicted?.["أفضل_الاحتمالات"] || [];
  const parsed = alternatives.map((item: { الفئة: string; درجة_غير_معايرة: number }) => {
    const [make, ...model] = String(item["الفئة"] || "").split("|");
    return { make, model: model.join("|"), confidence: item["درجة_غير_معايرة"] };
  });
  if (!predicted?.["الماركة"] || !predicted?.["الموديل"]) throw new Error("لم يستطع نموذج الهوية اقتراح سيارة لهذه الصورة.");
  return { make: predicted["الماركة"], model: predicted["الموديل"], confidence: predicted["درجة_غير_معايرة"] ?? 0, alternatives: parsed };
}
export async function estimateSale(details: Record<string, string>) {
  const result = await callLive<Record<string, any>>("predict_specs", [{
    make: details.make,
    model_std: details.model,
    year: Number(details.year),
    km_mileage: Number(details.mileage),
    trim_std: details.trim || "",
    city: details.city || "",
    transmission: details.transmission || "",
    type_fuel: details.fuel || "",
    condition: details.condition || "used",
  }]);
  const price = result["سعر_الإعلان_المقدر_ريال"];
  if (typeof price !== "number" || !Number.isFinite(price)) throw new Error("لم يُرجع نموذج المواصفات سعراً صالحاً.");
  return { price, notice: result.notice || result["تنبيه"] || "تقدير تجريبي لسعر الإعلان، وليس سعر بيع مضموناً." };
}
