"""
i18n/locale/registry.py
=======================

Registry of the languages the RVC WebUI can render in.

This module is *intentionally* not a JSON file because it carries
metadata that benefits from Python literals (display names, native
script names, and a script tag). It is imported lazily by
:mod:`i18n.i18n` via ``from i18n.locale.registry import LANGUAGES``.

Coverage
--------
The registry contains 99 locales spanning:

* The 11 locales already present in the original project (English,
  Chinese Simplified / Traditional / HK / SG, Japanese, French,
  Italian, Russian, Spanish, Turkish).
* 88 additional locales covering the most-spoken languages on every
  continent. Each new locale has a corresponding stub JSON file in
  ``i18n/locale/`` containing only the schema header; the new
  :class:`i18n.i18n.I18n` fallback chain (requested locale -> parent
  locale -> ``en_US`` -> source key) means untranslated strings fall
  back to English rather than to the Chinese source string.

Schema
------
Each entry is keyed by the locale code (``"en_US"``, ``"zh_CN"``) and
maps to a dict with three keys:

* ``name``    : English display name (used in the WebUI dropdown,
               sorted alphabetically).
* ``native``  : Native-script name (shown next to the English name
               so users can recognize their own language).
* ``script``  : ISO 15924 script code (``"Latn"``, ``"Hans"``,
               ``"Hant"``, ``"Cyrl"``, ``"Arab"``, ``"Hebr"``,
               ``"Deva"``, etc.) — used for font selection.

Adding a new language
---------------------
1. Add an entry to :data:`LANGUAGES` below.
2. Create ``i18n/locale/<code>.json`` with at least an empty dict
   ``{}`` (or fill in translations).
3. The WebUI language dropdown will automatically pick it up.
"""

LANGUAGES = {
    # --- Existing locales (translations present) ---------------------
    "en_US": {"name": "English (US)",       "native": "English",            "script": "Latn"},
    "zh_CN": {"name": "Chinese (Simplified)","native": "简体中文",          "script": "Hans"},
    "zh_TW": {"name": "Chinese (Traditional)","native": "繁體中文",          "script": "Hant"},
    "zh_HK": {"name": "Chinese (Hong Kong)","native": "繁體中文（香港）",     "script": "Hant"},
    "zh_SG": {"name": "Chinese (Singapore)","native": "简体中文（新加坡）",  "script": "Hans"},
    "ja_JP": {"name": "Japanese",           "native": "日本語",              "script": "Jpan"},
    "ko_KR": {"name": "Korean",             "native": "한국어",              "script": "Hang"},
    "fr_FR": {"name": "French (France)",   "native": "Français",           "script": "Latn"},
    "it_IT": {"name": "Italian",           "native": "Italiano",            "script": "Latn"},
    "ru_RU": {"name": "Russian",           "native": "Русский",            "script": "Cyrl"},
    "es_ES": {"name": "Spanish (Spain)",   "native": "Español (España)",    "script": "Latn"},
    "tr_TR": {"name": "Turkish",           "native": "Türkçe",              "script": "Latn"},

    # --- European languages ------------------------------------------
    "en_GB": {"name": "English (UK)",       "native": "English (UK)",        "script": "Latn"},
    "de_DE": {"name": "German",             "native": "Deutsch",            "script": "Latn"},
    "nl_NL": {"name": "Dutch",              "native": "Nederlands",         "script": "Latn"},
    "sv_SE": {"name": "Swedish",            "native": "Svenska",            "script": "Latn"},
    "no_NO": {"name": "Norwegian",          "native": "Norsk",              "script": "Latn"},
    "da_DK": {"name": "Danish",             "native": "Dansk",              "script": "Latn"},
    "fi_FI": {"name": "Finnish",            "native": "Suomi",              "script": "Latn"},
    "is_IS": {"name": "Icelandic",          "native": "Íslenska",           "script": "Latn"},
    "pl_PL": {"name": "Polish",             "native": "Polski",             "script": "Latn"},
    "cs_CZ": {"name": "Czech",              "native": "Čeština",            "script": "Latn"},
    "sk_SK": {"name": "Slovak",             "native": "Slovenčina",         "script": "Latn"},
    "hu_HU": {"name": "Hungarian",          "native": "Magyar",             "script": "Latn"},
    "ro_RO": {"name": "Romanian",           "native": "Română",             "script": "Latn"},
    "bg_BG": {"name": "Bulgarian",          "native": "Български",          "script": "Cyrl"},
    "sr_RS": {"name": "Serbian",            "native": "Српски",             "script": "Cyrl"},
    "hr_HR": {"name": "Croatian",            "native": "Hrvatski",           "script": "Latn"},
    "sl_SI": {"name": "Slovenian",          "native": "Slovenščina",        "script": "Latn"},
    "uk_UA": {"name": "Ukrainian",          "native": "Українська",         "script": "Cyrl"},
    "el_GR": {"name": "Greek",              "native": "Ελληνικά",           "script": "Grek"},
    "pt_PT": {"name": "Portuguese (Portugal)","native": "Português (Portugal)", "script": "Latn"},
    "pt_BR": {"name": "Portuguese (Brazil)","native": "Português (Brasil)", "script": "Latn"},
    "ca_ES": {"name": "Catalan",            "native": "Català",             "script": "Latn"},
    "eu_ES": {"name": "Basque",             "native": "Euskara",            "script": "Latn"},
    "gl_ES": {"name": "Galician",           "native": "Galego",             "script": "Latn"},
    "cy_GB": {"name": "Welsh",              "native": "Cymraeg",            "script": "Latn"},
    "ga_IE": {"name": "Irish",              "native": "Gaeilge",            "script": "Latn"},
    "et_EE": {"name": "Estonian",           "native": "Eesti",             "script": "Latn"},
    "lv_LV": {"name": "Latvian",            "native": "Latviešu",           "script": "Latn"},
    "lt_LT": {"name": "Lithuanian",         "native": "Lietuvių",           "script": "Latn"},
    "mt_MT": {"name": "Maltese",            "native": "Malti",             "script": "Latn"},

    # --- Asian languages ---------------------------------------------
    "id_ID": {"name": "Indonesian",         "native": "Bahasa Indonesia",   "script": "Latn"},
    "ms_MY": {"name": "Malay (Malaysia)",   "native": "Bahasa Melayu",       "script": "Latn"},
    "th_TH": {"name": "Thai",               "native": "ไทย",                "script": "Thai"},
    "vi_VN": {"name": "Vietnamese",         "native": "Tiếng Việt",         "script": "Latn"},
    "tl_PH": {"name": "Filipino (Tagalog)", "native": "Filipino",           "script": "Latn"},
    "km_KH": {"name": "Khmer",              "native": "ខ្មែរ",               "script": "Khmr"},
    "lo_LA": {"name": "Lao",                "native": "ລາວ",                "script": "Laoo"},
    "my_MM": {"name": "Burmese",            "native": "မြန်မာ",              "script": "Mymr"},
    "hi_IN": {"name": "Hindi",              "native": "हिन्दी",              "script": "Deva"},
    "bn_IN": {"name": "Bengali (India)",    "native": "বাংলা",              "script": "Beng"},
    "bn_BD": {"name": "Bengali (Bangladesh)","native": "বাংলা",             "script": "Beng"},
    "ta_IN": {"name": "Tamil (India)",      "native": "தமிழ்",               "script": "Taml"},
    "te_IN": {"name": "Telugu",             "native": "తెలుగు",             "script": "Telu"},
    "kn_IN": {"name": "Kannada",            "native": "ಕನ್ನಡ",              "script": "Knda"},
    "ml_IN": {"name": "Malayalam",          "native": "മലയാളം",            "script": "Mlym"},
    "mr_IN": {"name": "Marathi",            "native": "मराठी",               "script": "Deva"},
    "gu_IN": {"name": "Gujarati",           "native": "ગુજરાતી",            "script": "Gujr"},
    "pa_IN": {"name": "Punjabi (Gurmukhi)", "native": "ਪੰਜਾਬੀ",              "script": "Guru"},
    "or_IN": {"name": "Odia",               "native": "ଓଡ଼ିଆ",             "script": "Orya"},
    "as_IN": {"name": "Assamese",           "native": "অসমীয়া",            "script": "Beng"},
    "si_LK": {"name": "Sinhala",            "native": "සිංහල",             "script": "Sinh"},
    "ne_NP": {"name": "Nepali",             "native": "नेपाली",              "script": "Deva"},
    "ur_PK": {"name": "Urdu (Pakistan)",    "native": "اُردُو",              "script": "Arab"},
    "ur_IN": {"name": "Urdu (India)",       "native": "اُردُو",              "script": "Arab"},
    "fa_IR": {"name": "Persian (Farsi)",    "native": "فارسی",              "script": "Arab"},
    "ps_AF": {"name": "Pashto",             "native": "پښتو",               "script": "Arab"},
    "dv_MV": {"name": "Dhivehi",            "native": "ދިވެހި",             "script": "Thaa"},

    # --- Middle East / West Asia -------------------------------------
    "ar_SA": {"name": "Arabic (Saudi Arabia)","native": "العربية",          "script": "Arab"},
    "ar_EG": {"name": "Arabic (Egypt)",     "native": "العربية (مصر)",      "script": "Arab"},
    "ar_AE": {"name": "Arabic (UAE)",       "native": "العربية (الإمارات)", "script": "Arab"},
    "he_IL": {"name": "Hebrew",             "native": "עברית",               "script": "Hebr"},
    "ku_TR": {"name": "Kurdish (Kurmanji)", "native": "Kurdî",              "script": "Latn"},
    "az_AZ": {"name": "Azerbaijani",        "native": "Azərbaycanca",       "script": "Latn"},
    "ka_GE": {"name": "Georgian",           "native": "ქართული",            "script": "Geor"},
    "hy_AM": {"name": "Armenian",           "native": "Հայերեն",            "script": "Armn"},

    # --- Africa -------------------------------------------------------
    "sw_KE": {"name": "Swahili (Kenya)",    "native": "Kiswahili (Kenya)",  "script": "Latn"},
    "sw_TZ": {"name": "Swahili (Tanzania)", "native": "Kiswahili (Tanzania)","script": "Latn"},
    "am_ET": {"name": "Amharic",            "native": "አማርኛ",                "script": "Ethi"},
    "ha_NG": {"name": "Hausa",              "native": "Hausa",               "script": "Latn"},
    "yo_NG": {"name": "Yoruba",             "native": "Yorùbá",             "script": "Latn"},
    "ig_NG": {"name": "Igbo",               "native": "Igbo",               "script": "Latn"},
    "zu_ZA": {"name": "Zulu",               "native": "isiZulu",            "script": "Latn"},
    "xh_ZA": {"name": "Xhosa",              "native": "isiXhosa",           "script": "Latn"},
    "af_ZA": {"name": "Afrikaans",          "native": "Afrikaans",          "script": "Latn"},
    "om_ET": {"name": "Oromo",              "native": "Oromoo",             "script": "Latn"},
    "sn_ZW": {"name": "Shona",              "native": "ChiShona",           "script": "Latn"},
    "st_ZA": {"name": "Southern Sotho",     "native": "Sesotho",            "script": "Latn"},

    # --- Central Asia / Siberia --------------------------------------
    "kk_KZ": {"name": "Kazakh",             "native": "Қазақша",            "script": "Cyrl"},
    "ky_KG": {"name": "Kyrgyz",             "native": "Кыргызча",           "script": "Cyrl"},
    "uz_UZ": {"name": "Uzbek (Latin)",      "native": "Oʻzbekcha",          "script": "Latn"},
    "tg_TJ": {"name": "Tajik",              "native": "Тоҷикӣ",             "script": "Cyrl"},
    "tk_TM": {"name": "Turkmen",            "native": "Türkmen",            "script": "Latn"},
    "tt_RU": {"name": "Tatar",              "native": "Татарча",            "script": "Cyrl"},
    "mn_MN": {"name": "Mongolian (Cyrillic)","native": "Монгол",            "script": "Cyrl"},
    "mn_CN": {"name": "Mongolian (Inner Mongolia)","native": "ᠮᠣᠩᠭᠣᠯ",     "script": "Mong"},

    # --- South-East Asia / Pacific -----------------------------------
    "jv_ID": {"name": "Javanese (Latin)",   "native": "Basa Jawa",          "script": "Latn"},
    "su_ID": {"name": "Sundanese",          "native": "Basa Sunda",        "script": "Latn"},
    "ms_ID": {"name": "Malay (Indonesia)",  "native": "Bahasa Melayu (Indonesia)","script": "Latn"},
    "mi_NZ": {"name": "Maori",              "native": "Te Reo Māori",       "script": "Latn"},
    "haw_US": {"name": "Hawaiian",          "native": "ʻŌlelo Hawaiʻi",      "script": "Latn"},

    # --- Misc ---------------------------------------------------------
    "es_MX": {"name": "Spanish (Mexico)",   "native": "Español (México)",   "script": "Latn"},
    "fr_CA": {"name": "French (Canada)",    "native": "Français (Canada)",  "script": "Latn"},
    "nl_BE": {"name": "Dutch (Belgium)",    "native": "Nederlands (België)","script": "Latn"},
    "de_AT": {"name": "German (Austria)",   "native": "Deutsch (Österreich)","script": "Latn"},
    "de_CH": {"name": "German (Switzerland)","native": "Deutsch (Schweiz)",  "script": "Latn"},
    "it_CH": {"name": "Italian (Switzerland)","native": "Italiano (Svizzera)","script": "Latn"},
    "en_AU": {"name": "English (Australia)","native": "English (Australia)","script": "Latn"},
    "en_CA": {"name": "English (Canada)",   "native": "English (Canada)",   "script": "Latn"},
    "en_IN": {"name": "English (India)",    "native": "English (India)",    "script": "Latn"},
    "en_ZA": {"name": "English (South Africa)","native": "English (South Africa)","script": "Latn"},
    "val_ES": {"name": "Valencian",          "native": "Valencià",          "script": "Latn"},
    "fo_FO": {"name": "Faroese",             "native": "Føroyskt",          "script": "Latn"},
    "sa_IN": {"name": "Sanskrit",            "native": "संस्कृतम्",          "script": "Deva"},
}

# Sanity check at import time: the registry must contain at least the
# 11 locales already shipped by the original project so that existing
# installations keep working unchanged.
_REQUIRED_EXISTING = frozenset({
    "en_US", "zh_CN", "zh_TW", "zh_HK", "zh_SG",
    "ja_JP", "fr_FR", "it_IT", "ru_RU", "es_ES", "tr_TR",
})
_missing = _REQUIRED_EXISTING - set(LANGUAGES)
if _missing:  # pragma: no cover - defensive
    raise RuntimeError(
        f"i18n.locale.registry.LANGUAGES is missing required locales: "
        f"{sorted(_missing)}"
    )
