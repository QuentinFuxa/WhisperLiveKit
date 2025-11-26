# Transcription: Supported Language

WLK supports transcription in the following languages:

| ISO Code | Language Name        |
|----------|---------------------|
| en       | English             |
| zh       | Chinese             |
| de       | German              |
| es       | Spanish             |
| ru       | Russian             |
| ko       | Korean              |
| fr       | French              |
| ja       | Japanese            |
| pt       | Portuguese          |
| tr       | Turkish             |
| pl       | Polish              |
| ca       | Catalan             |
| nl       | Dutch               |
| ar       | Arabic              |
| sv       | Swedish             |
| it       | Italian             |
| id       | Indonesian          |
| hi       | Hindi               |
| fi       | Finnish             |
| vi       | Vietnamese          |
| he       | Hebrew              |
| uk       | Ukrainian           |
| el       | Greek               |
| ms       | Malay               |
| cs       | Czech               |
| ro       | Romanian            |
| da       | Danish              |
| hu       | Hungarian           |
| ta       | Tamil               |
| no       | Norwegian           |
| th       | Thai                |
| ur       | Urdu                |
| hr       | Croatian            |
| bg       | Bulgarian           |
| lt       | Lithuanian          |
| la       | Latin               |
| mi       | Maori               |
| ml       | Malayalam           |
| cy       | Welsh               |
| sk       | Slovak              |
| te       | Telugu              |
| fa       | Persian             |
| lv       | Latvian             |
| bn       | Bengali             |
| sr       | Serbian             |
| az       | Azerbaijani         |
| sl       | Slovenian           |
| kn       | Kannada             |
| et       | Estonian            |
| mk       | Macedonian          |
| br       | Breton              |
| eu       | Basque              |
| is       | Icelandic           |
| hy       | Armenian            |
| ne       | Nepali              |
| mn       | Mongolian           |
| bs       | Bosnian             |
| kk       | Kazakh              |
| sq       | Albanian            |
| sw       | Swahili             |
| gl       | Galician            |
| mr       | Marathi             |
| pa       | Punjabi             |
| si       | Sinhala             |
| km       | Khmer               |
| sn       | Shona               |
| yo       | Yoruba              |
| so       | Somali              |
| af       | Afrikaans           |
| oc       | Occitan             |
| ka       | Georgian            |
| be       | Belarusian          |
| tg       | Tajik               |
| sd       | Sindhi              |
| gu       | Gujarati            |
| am       | Amharic             |
| yi       | Yiddish             |
| lo       | Lao                 |
| uz       | Uzbek               |
| fo       | Faroese             |
| ht       | Haitian Creole      |
| ps       | Pashto              |
| tk       | Turkmen             |
| nn       | Nynorsk             |
| mt       | Maltese             |
| sa       | Sanskrit            |
| lb       | Luxembourgish       |
| my       | Myanmar             |
| bo       | Tibetan             |
| tl       | Tagalog             |
| mg       | Malagasy            |
| as       | Assamese            |
| tt       | Tatar               |
| haw      | Hawaiian            |
| ln       | Lingala             |
| ha       | Hausa               |
| ba       | Bashkir             |
| jw       | Javanese            |
| su       | Sundanese           |
| yue      | Cantonese           |


# Translation: Supported Languages 

WLK supports translation into **201 languages** from the FLORES-200 dataset through the [NLLW](https://github.com/QuentinFuxa/NoLanguageLeftWaiting) translation system. 

## How to Specify Languages

You can specify languages in **three different ways**:

1. **Language Name** (case-insensitive): `"English"`, `"French"`, `"Spanish"`
2. **ISO Language Code**: `"en"`, `"fr"`, `"es"`
3. **NLLB Code** (FLORES-200): `"eng_Latn"`, `"fra_Latn"`, `"spa_Latn"`

## Usage Examples

### Command Line
```bash
# Using language name
whisperlivekit-server --target-language "French"

# Using ISO code
whisperlivekit-server --target-language fr

# Using NLLB code
whisperlivekit-server --target-language fra_Latn
```

### Python API
```python
from nllw.translation import get_language_info

# Get language information by name
lang_info = get_language_info("French")
print(lang_info)
# {'name': 'French', 'nllb': 'fra_Latn', 'language_code': 'fr'}

# Get language information by ISO code
lang_info = get_language_info("fr")

# Get language information by NLLB code
lang_info = get_language_info("fra_Latn")

# All three return the same result
```

## Complete Language List

The following table lists all 201 supported languages with their corresponding codes:

| Language Name | ISO Code | NLLB Code |
|---------------|----------|-----------|
| Acehnese (Arabic script) | ace_Arab | ace_Arab |
| Acehnese (Latin script) | ace_Latn | ace_Latn |
| Mesopotamian Arabic | acm_Arab | acm_Arab |
| Ta'izzi-Adeni Arabic | acq_Arab | acq_Arab |
| Tunisian Arabic | aeb_Arab | aeb_Arab |
| Afrikaans | af | afr_Latn |
| South Levantine Arabic | ajp_Arab | ajp_Arab |
| Akan | ak | aka_Latn |
| Tosk Albanian | als | als_Latn |
| Amharic | am | amh_Ethi |
| North Levantine Arabic | apc_Arab | apc_Arab |
| Modern Standard Arabic | ar | arb_Arab |
| Modern Standard Arabic (Romanized) | arb_Latn | arb_Latn |
| Najdi Arabic | ars_Arab | ars_Arab |
| Moroccan Arabic | ary_Arab | ary_Arab |
| Egyptian Arabic | arz_Arab | arz_Arab |
| Assamese | as | asm_Beng |
| Asturian | ast | ast_Latn |
| Awadhi | awa | awa_Deva |
| Central Aymara | ay | ayr_Latn |
| South Azerbaijani | azb | azb_Arab |
| North Azerbaijani | az | azj_Latn |
| Bashkir | ba | bak_Cyrl |
| Bambara | bm | bam_Latn |
| Balinese | ban | ban_Latn |
| Belarusian | be | bel_Cyrl |
| Bemba | bem | bem_Latn |
| Bengali | bn | ben_Beng |
| Bhojpuri | bho | bho_Deva |
| Banjar (Arabic script) | bjn_Arab | bjn_Arab |
| Banjar (Latin script) | bjn_Latn | bjn_Latn |
| Standard Tibetan | bo | bod_Tibt |
| Bosnian | bs | bos_Latn |
| Buginese | bug | bug_Latn |
| Bulgarian | bg | bul_Cyrl |
| Catalan | ca | cat_Latn |
| Cebuano | ceb | ceb_Latn |
| Czech | cs | ces_Latn |
| Chokwe | cjk | cjk_Latn |
| Central Kurdish | ckb | ckb_Arab |
| Crimean Tatar | crh | crh_Latn |
| Welsh | cy | cym_Latn |
| Danish | da | dan_Latn |
| German | de | deu_Latn |
| Southwestern Dinka | dik | dik_Latn |
| Dyula | dyu | dyu_Latn |
| Dzongkha | dz | dzo_Tibt |
| Greek | el | ell_Grek |
| English | en | eng_Latn |
| Esperanto | eo | epo_Latn |
| Estonian | et | est_Latn |
| Basque | eu | eus_Latn |
| Ewe | ee | ewe_Latn |
| Faroese | fo | fao_Latn |
| Fijian | fj | fij_Latn |
| Finnish | fi | fin_Latn |
| Fon | fon | fon_Latn |
| French | fr | fra_Latn |
| Friulian | fur-IT | fur_Latn |
| Nigerian Fulfulde | fuv | fuv_Latn |
| West Central Oromo | om | gaz_Latn |
| Scottish Gaelic | gd | gla_Latn |
| Irish | ga-IE | gle_Latn |
| Galician | gl | glg_Latn |
| Guarani | gn | grn_Latn |
| Gujarati | gu-IN | guj_Gujr |
| Haitian Creole | ht | hat_Latn |
| Hausa | ha | hau_Latn |
| Hebrew | he | heb_Hebr |
| Hindi | hi | hin_Deva |
| Chhattisgarhi | hne | hne_Deva |
| Croatian | hr | hrv_Latn |
| Hungarian | hu | hun_Latn |
| Armenian | hy-AM | hye_Armn |
| Igbo | ig | ibo_Latn |
| Ilocano | ilo | ilo_Latn |
| Indonesian | id | ind_Latn |
| Icelandic | is | isl_Latn |
| Italian | it | ita_Latn |
| Javanese | jv | jav_Latn |
| Japanese | ja | jpn_Jpan |
| Kabyle | kab | kab_Latn |
| Jingpho | kac | kac_Latn |
| Kamba | kam | kam_Latn |
| Kannada | kn | kan_Knda |
| Kashmiri (Arabic script) | kas_Arab | kas_Arab |
| Kashmiri (Devanagari script) | kas_Deva | kas_Deva |
| Georgian | ka | kat_Geor |
| Kazakh | kk | kaz_Cyrl |
| Kabiyè | kbp | kbp_Latn |
| Kabuverdianu | kea | kea_Latn |
| Halh Mongolian | mn | khk_Cyrl |
| Khmer | km | khm_Khmr |
| Kikuyu | ki | kik_Latn |
| Kinyarwanda | rw | kin_Latn |
| Kyrgyz | ky | kir_Cyrl |
| Kimbundu | kmb | kmb_Latn |
| Northern Kurdish | kmr | kmr_Latn |
| Central Kanuri (Arabic script) | knc_Arab | knc_Arab |
| Central Kanuri (Latin script) | knc_Latn | knc_Latn |
| Kikongo | kg | kon_Latn |
| Korean | ko | kor_Hang |
| Lao | lo | lao_Laoo |
| Ligurian | lij | lij_Latn |
| Limburgish | li | lim_Latn |
| Lingala | ln | lin_Latn |
| Lithuanian | lt | lit_Latn |
| Lombard | lmo | lmo_Latn |
| Latgalian | ltg | ltg_Latn |
| Luxembourgish | lb | ltz_Latn |
| Luba-Kasai | lua | lua_Latn |
| Ganda | lg | lug_Latn |
| Luo | luo | luo_Latn |
| Mizo | lus | lus_Latn |
| Standard Latvian | lv | lvs_Latn |
| Magahi | mag | mag_Deva |
| Maithili | mai | mai_Deva |
| Malayalam | ml-IN | mal_Mlym |
| Marathi | mr | mar_Deva |
| Minangkabau (Arabic script) | min_Arab | min_Arab |
| Minangkabau (Latin script) | min_Latn | min_Latn |
| Macedonian | mk | mkd_Cyrl |
| Maltese | mt | mlt_Latn |
| Meitei (Bengali script) | mni | mni_Beng |
| Mossi | mos | mos_Latn |
| Maori | mi | mri_Latn |
| Burmese | my | mya_Mymr |
| Dutch | nl | nld_Latn |
| Norwegian Nynorsk | nn-NO | nno_Latn |
| Norwegian Bokmål | nb | nob_Latn |
| Nepali | ne-NP | npi_Deva |
| Northern Sotho | nso | nso_Latn |
| Nuer | nus | nus_Latn |
| Nyanja | ny | nya_Latn |
| Occitan | oc | oci_Latn |
| Odia | or | ory_Orya |
| Pangasinan | pag | pag_Latn |
| Eastern Panjabi | pa | pan_Guru |
| Papiamento | pap | pap_Latn |
| Southern Pashto | pbt | pbt_Arab |
| Western Persian | fa | pes_Arab |
| Plateau Malagasy | mg | plt_Latn |
| Polish | pl | pol_Latn |
| Portuguese | pt-PT | por_Latn |
| Dari | fa-AF | prs_Arab |
| Ayacucho Quechua | qu | quy_Latn |
| Romanian | ro | ron_Latn |
| Rundi | rn | run_Latn |
| Russian | ru | rus_Cyrl |
| Sango | sg | sag_Latn |
| Sanskrit | sa | san_Deva |
| Santali | sat | sat_Olck |
| Sicilian | scn | scn_Latn |
| Shan | shn | shn_Mymr |
| Sinhala | si-LK | sin_Sinh |
| Slovak | sk | slk_Latn |
| Slovenian | sl | slv_Latn |
| Samoan | sm | smo_Latn |
| Shona | sn | sna_Latn |
| Sindhi | sd | snd_Arab |
| Somali | so | som_Latn |
| Southern Sotho | st | sot_Latn |
| Spanish | es-ES | spa_Latn |
| Sardinian | sc | srd_Latn |
| Serbian | sr | srp_Cyrl |
| Swati | ss | ssw_Latn |
| Sundanese | su | sun_Latn |
| Swedish | sv-SE | swe_Latn |
| Swahili | sw | swh_Latn |
| Silesian | szl | szl_Latn |
| Tamil | ta | tam_Taml |
| Tamasheq (Latin script) | taq_Latn | taq_Latn |
| Tamasheq (Tifinagh script) | taq_Tfng | taq_Tfng |
| Tatar | tt-RU | tat_Cyrl |
| Telugu | te | tel_Telu |
| Tajik | tg | tgk_Cyrl |
| Tagalog | tl | tgl_Latn |
| Thai | th | tha_Thai |
| Tigrinya | ti | tir_Ethi |
| Tok Pisin | tpi | tpi_Latn |
| Tswana | tn | tsn_Latn |
| Tsonga | ts | tso_Latn |
| Turkmen | tk | tuk_Latn |
| Tumbuka | tum | tum_Latn |
| Turkish | tr | tur_Latn |
| Twi | tw | twi_Latn |
| Central Atlas Tamazight | tzm | tzm_Tfng |
| Uyghur | ug | uig_Arab |
| Ukrainian | uk | ukr_Cyrl |
| Umbundu | umb | umb_Latn |
| Urdu | ur | urd_Arab |
| Northern Uzbek | uz | uzn_Latn |
| Venetian | vec | vec_Latn |
| Vietnamese | vi | vie_Latn |
| Waray | war | war_Latn |
| Wolof | wo | wol_Latn |
| Xhosa | xh | xho_Latn |
| Eastern Yiddish | yi | ydd_Hebr |
| Yoruba | yo | yor_Latn |
| Yue Chinese | yue | yue_Hant |
| Chinese (Simplified) | zh-CN | zho_Hans |
| Chinese (Traditional) | zh-TW | zho_Hant |
| Standard Malay | ms | zsm_Latn |
| Zulu | zu | zul_Latn |

## Special Features

### Multiple Script Support
Several languages are available in multiple scripts (e.g., Arabic and Latin):
- **Acehnese**: Arabic (`ace_Arab`) and Latin (`ace_Latn`)
- **Banjar**: Arabic (`bjn_Arab`) and Latin (`bjn_Latn`)
- **Kashmiri**: Arabic (`kas_Arab`) and Devanagari (`kas_Deva`)
- **Minangkabau**: Arabic (`min_Arab`) and Latin (`min_Latn`)
- **Tamasheq**: Latin (`taq_Latn`) and Tifinagh (`taq_Tfng`)
- **Central Kanuri**: Arabic (`knc_Arab`) and Latin (`knc_Latn`)