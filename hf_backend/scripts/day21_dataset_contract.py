"""أعلام منفصلة ومسارات تعرف وهوية للبيانات؛ لا تدريب."""
import hashlib
import json
from day20_common import read_csv, sha256
from day20_data_policy import yes

FLAG_COLUMNS = {
    "flag_source_unavailable": "recovery_source_redirected_or_unavailable",
    "flag_high_annual_mileage": "annualized_mileage_proxy_high_review",
    "flag_price_iqr_outlier": "global_price_iqr_outlier_review",
    "flag_mileage_exact_100": "mileage_exact_100_review",
    "flag_young_low_mileage": "young_used_low_mileage_review",
}


def separate_flags(row):
    flags = set(row.get("data_quality_flags", "").split(";"))
    return {column: code in flags for column, code in FLAG_COLUMNS.items()}


def recognition_route(row):
    if yes(row.get("usable_recognition")): return "model_and_make"
    if yes(row.get("usable_make_recognition")): return "make_only"
    return "unavailable"


def split_fingerprint(rows):
    values = sorted((r["listing_uid"], r["group_id"], r["split"]) for r in rows)
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def dataset_manifest(processed, version):
    hashes = {name: sha256(processed / name) for name in ("listings.csv", "images.csv", "schema.json")}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"dataset_version": version, "dataset_fingerprint": fingerprint,
            "split_fingerprint": split_fingerprint(read_csv(processed / "listings.csv")), "input_files_sha256": hashes,
            "required_model_metadata": ["dataset_version", "dataset_fingerprint", "split_fingerprint"],
            "training_status": "not_started", "legacy_metrics_are_current_evaluation": False}


def assert_artifact_compatible(metadata, manifest):
    """يستدعيه تحميل نموذج مستقبلًا؛ المقاييس القديمة بلا هوية لا تعتمد تلقائيًا."""
    for field in ("dataset_version", "dataset_fingerprint", "split_fingerprint"):
        if not metadata.get(field) or metadata[field] != manifest[field]:
            raise ValueError("النموذج لا يطابق هوية البيانات الحالية: " + field)
    return True
