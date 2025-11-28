## 1. PRD – Product Requirements Document

### 1.1 Základní informace

- **Pracovní název produktu:** `NÁZEV DOPLŇ`
- **Typ produktu:** B2B SaaS / on-prem systém pro řízení životního cyklu dokumentů
- **Cíl:**

  - sjednotit tok firemních dokumentů (faktury, smlouvy, objednávky, interní dokumenty)
  - snížit chaos v e-mailech / šanonech
  - zrychlit a zprůhlednit schvalování
  - mít auditovatelný, právně „uchopitelný“ archiv

### 1.2 Proč produkt vzniká (problém)

**Typické problémy firem:**

- dokumenty chodí do různých schránek (osobní e-mail, info@, datová schránka, papír na recepci)
- nikdo nemá celkový přehled „co kde je“
- schvalování probíhá přes e-maily nebo chat → ztrácí se historie a zodpovědnost
- dokumenty končí v šanonech bez metadata → těžké dohledávání („kde je ta smlouva z 2021?“)
- audit / kontrola je peklo (chybí logy, kdo co kdy schválil)

**Jak to produkt řeší:**

- jeden vstupní bod pro dokumenty (sběr z kanálů)
- jednotné workflow schvalování (definovaná pravidla, role)
- bezpečný elektronický archiv s vyhledáváním
- auditní stopa (kdo co kdy změnil/schválil/viděl)

### 1.3 Cíloví uživatelé / persony

1. **Účetní / finanční oddělení**

   - řeší příchozí faktury, dobropisy, objednávky
   - potřebuje schválit částky, přiřadit střediska, exportovat do účetnictví

2. **Back-office / office manažer**

   - zpracovává poštu, skenuje dokumenty, zakládá do systému
   - potřebuje jednoduché UI, minimum klikání

3. **Manažer / schvalovatel**

   - přijde mu „úkol ke schválení“
   - potřebuje rychle vidět, o co jde, a rozhodnout: schválit / vrátit / zamítnout

4. **Management / kontrolor**

   - potřebuje reporty, dohledávání, auditní logy
   - hledá podle firmy, částky, období, typu dokumentu…

### 1.4 High-level cíle produktu (MVP)

**Business cíle (první rok):**

- zkrátit dobu schválení faktury (příchod → schválení) min. o 30 %
- snížit počet „ztracených“ / nedohledatelných dokumentů prakticky na 0
- umožnit onboardovat novou firmu do 1–2 dnů bez custom vývoje

**Produktové cíle (MVP):**

- mít funkční end-to-end tok:
  příjem dokumentu → extrakce dat → schválení → archivace
- zvládnout min. 3 základní zdroje dokumentů:

  - e-mail
  - ruční upload / drag-and-drop
  - skenované PDF

- jednoduchý konfigurovatelný schvalovací workflow (bez developer zásahu)

---

### 1.5 Funkční požadavky (MVP)

#### 1.5.1 Příjem dokumentů

- **Zdroje:**

  - IMAP/SMTP konektor pro e-mailové schránky (např. [invoices@firma.cz](mailto:invoices@firma.cz))
  - ruční nahrání souboru (PDF, obrázky)
  - možnost později přidat:

    - datová schránka
    - SFTP / API konektor z jiného systému

- **Požadavky:**

  - automatické přiřazení zdroje (channel) k dokumentu
  - zobrazení fronty „nové dokumenty ke zpracování“

#### 1.5.2 Rozpoznání / vytěžení dat

- použít OCR (pokud je to sken)
- pokus o základní vytěžení klíčových polí:

  - datum, číslo dokladu, protistrana, částka, měna

- možnost ruční korekce uživatelem
- logovat úspěšnost (pro další trénink / zlepšení)

#### 1.5.3 Workflow schvalování

- definice schvalovacích pravidel:

  - podle částky
  - podle typu dokumentu
  - podle střediska / nákladového místa

- funkce:

  - vytvoření schvalovacího schématu (např. „Faktury do 50k“ → účetní → vedoucí střediska)
  - notifikace schvalovateli (e-mail + v aplikaci)
  - akce: **schválit / zamítnout / vrátit k doplnění**
  - komentáře k dokumentu
  - historie změn ve workflow

#### 1.5.4 Archiv a vyhledávání

- bezpečné uložení originálního souboru (read-only)
- metadata:

  - typ dokumentu
  - firma / protistrana
  - částka, měna, datum
  - stav schválení

- vyhledávání:

  - fulltext v metadatech
  - filtry (datum, protistrana, stav, typ dokumentu)

#### 1.5.5 Uživatelské role a oprávnění

- **Role:**

  - Admin (nastavení, správa uživatelů, schémata)
  - Zpracovatel (přiřazuje dokumenty, opravuje data)
  - Schvalovatel
  - Pouze čtení (např. auditor)

- **Požadavky:**

  - RBAC (role-based access control)
  - omezení přístupu k dokumentům dle role a firmy (multi-tenant)

---

### 1.6 Nefunkční požadavky

- **Bezpečnost:**

  - šifrování dat v klidu (at-rest) i při přenosu (TLS)
  - auditní log (přístupy, změny, schválení)
  - možnost SSO (SAML/OIDC) – ne nutně v MVP, ale v roadmapě

- **Výkon:**

  - systém musí zvládnout tisíce dokumentů měsíčně / tenant
  - odezva pro běžné operace < 1–2 s (zobrazení detailu, vyhledávání v běžných datech)

- **Dostupnost:**

  - cílení na 99,5+ % uptime (SaaS)

- **Právní / compliance:**

  - možnost definovat retenční politiku (jak dlouho se dokumenty uchovávají)
  - export dat a dokumentů (pro audit / změnu dodavatele)

---

### 1.7 Out of scope (pro MVP)

- plná DMS náhrada pro všechny typy souborů (CAD, videa, velké binárky…)
- právní kvalifikované podpisy a časová razítka (může být další modul)
- hluboká integrace do konkrétních ERP (JDE, SAP, Helios…) – v MVP jen export/import přes soubory/API
- mobilní aplikace (nativní) – pouze responzivní web

---

## 2. Roadmapa

Časové jednotky si můžeš přepočítat na měsíce/čtvrtletí, nechávám to záměrně generické.

### Fáze 0 – Discovery & návrh (cca 2–4 týdny)

- rozhovory s klíčovými zákazníky / pilotními firmami
- validace hlavního use-case:

  - příjem faktur + schvalování + archiv

- návrh datového modelu (dokument, verze, metadata, workflow, tenant)
- rozhodnutí: SaaS pouze vs. i on-prem varianta

**Výstup:**

- potvrzený rozsah MVP
- high-level architektura
- seznam pilotních zákazníků

---

### Fáze 1 – MVP jádro (cca 2–3 měsíce)

**Cíl:** mít funkční end-to-end tok pro pilotní zákazníky.

- Implementace:

  - přihlášení, základní RBAC
  - ruční upload dokumentů
  - jednoduché workflow schválení (1–2 schvalovatelé)
  - archiv + základní vyhledávání

- Admin UI:

  - nastavení typů dokumentů
  - nastavení schvalovacích schémat

**Milník:**

- pilotní zákazník může zpracovávat faktury plně v systému.

---

### Fáze 2 – Integrace & automatizace (cca 2–3 měsíce)

- konektor na e-mail (IMAP) pro automatický sběr dokumentů
- základní OCR a vytěžení dat
- inteligentnější workflow:

  - pravidla podle částky / střediska

- notifikace (e-mail, notifikační centrum v aplikaci)

**Milník:**

- > 80 % příchozích faktur vstupuje do systému bez ručního uploadu.

---

### Fáze 3 – Rozšíření funkcí a reporting (cca 2–3 měsíce)

- pokročilé vyhledávání (vícekriteriální, uložení filtrů)
- auditní log s UI pro auditora
- základní dashboard:

  - počet dokumentů / měsíc
  - délka schvalovacího cyklu

- exporty pro účetnictví (CSV/Excel/API)

---

### Fáze 4 – Enterprise features (střednědobě)

- SSO (SAML, OIDC)
- granulárnější práva (na úrovni střediska, typu dokumentu, tenant-level)
- API pro integraci s ERP/CRM
- multi-regionální hosting (dle poptávky zákazníků)

---

## 3. Poznámky, rizika, otevřené otázky

### 3.1 Rizika

- **Právní / známkové:**

  - brand musí být čistý neologismus (jak psal právník), jinak hrozí námitky / zaměnitelnost

- **Change management u zákazníka:**

  - firmy jsou zvyklé na e-maily a papír → je potřeba onboarding, školení, možná konzultační balíčky

- **Integrace:**

  - zákazníci budou dříve či později chtít „to napojit na jejich ERP/účetnictví“ → riziko rozsahu custom prací

### 3.2 Otevřené otázky (co bude potřeba rozhodnout)

- Bude produkt **čistě SaaS**, nebo i **on-prem** (pro banky, úřady)?
- Jaký je **pricing model**:

  - per uživatel
  - per dokument / měsíc
  - per „balík“ (plán S/M/L)?

- Jaká je **minimální verze auditu**, kterou musíme splnit (např. požadavky konkrétních regulovaných zákazníků)?
- Jak moc chceme jít do **automatizace vytěžování** (jen basic OCR vs. propojení na specializovaný engine / službu)?

### 3.3 Interní poznámky pro další iteraci

- Až budeš mít finální název společnosti / produktu:

  - doplnit do dokumentu všude `NÁZEV DOPLŇ`
  - připravit krátkou **brand větu** (1–2 věty) pro úvod PRD

- Doporučuje se vést k tomu:

  - **technickou dokumentaci** (architektura, datový model),
  - **UX specifikaci** (wireframy, user flows),
  - **implementační backlog** (user stories v JIRA / Linear / YouTrack).
