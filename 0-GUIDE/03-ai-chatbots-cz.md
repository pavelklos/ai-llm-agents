# Průvodce kurzem Vývoj AI chatbotů

Vytvářejte vlastní AI chatboty s pamětí, osobností a kontextovými odpověďmi pomocí Pythonu a moderních AI frameworků.

## Výsledky učení

Po dokončení tohoto kurzu budete schopni:
- Vytvářet vlastní AI asistenty
- Integrovat API a databáze
- Pracovat s daty a RAG (Retrieval-Augmented Generation)
- Simulovat osobnosti a rozhodovací logiku

## Cílová skupina

### Tradiční vývojáři
Integrace AI nástrojů do aplikací pomocí LangChain a GPT API, propojení asistentů s databázemi a externími službami.

### AI specialisté
Pokročilé nástroje pro vytváření a ladění AI asistentů, vektorové databáze, RAG a znalostní grafy.

### Podnikatelé
Zlepšení zákaznické zkušenosti pomocí chatbotů, nasazení GPT asistentů pro podporu a komunikaci.

---

## 01. Úvod do AI asistentů a vytvoření prvního GPT asistenta

Naučte se základy AI asistentů a vytvořte základního konverzačního agenta.

```python
import openai
from openai import OpenAI

class ZakladniGPTAsistent:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.historie_konverzace = []
    
    def chat(self, zprava_uzivatele):
        self.historie_konverzace.append({"role": "user", "content": zprava_uzivatele})
        
        odpoved = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=self.historie_konverzace,
            max_tokens=150
        )
        
        odpoved_asistenta = odpoved.choices[0].message.content
        self.historie_konverzace.append({"role": "assistant", "content": odpoved_asistenta})
        
        return odpoved_asistenta

# Použití
asistent = ZakladniGPTAsistent("vas-api-klic")
odpoved = asistent.chat("Ahoj, jak mi můžeš pomoci?")
print(odpoved)
```

## 02. Schopnosti a omezení GPT asistentů

Pochopte silné stránky, slabiny a vhodné případy použití pro GPT asistenty. Naučte se strategie prompt engineeringu.

```python
class PromptInzenier:
    def __init__(self):
        self.systemove_prompty = {
            "uzitecny": "Jsi užitečný asistent, který poskytuje jasné a stručné odpovědi.",
            "kreativni": "Jsi kreativní asistent pro psaní, který pomáhá s vyprávěním příběhů.",
            "technicky": "Jsi technický expert, který jednoduše vysvětluje složité koncepty."
        }
    
    def vytvor_prompt(self, role, kontext, dotaz_uzivatele):
        return f"""
        {self.systemove_prompty.get(role, self.systemove_prompty['uzitecny'])}
        
        Kontext: {kontext}
        
        Dotaz uživatele: {dotaz_uzivatele}
        
        Prosím, poskytni odpověď, která je vhodná pro daný kontext.
        """

# Použití
prompt_ing = PromptInzenier()
prompt = prompt_ing.vytvor_prompt("technicky", "Programování v Pythonu", "Jak fungují cykly?")
```

## 03. Vektorové databáze a jejich aplikace

Implementujte vektorové úložiště a vyhledávání podobnosti pro odpovídání na otázky založené na dokumentech pomocí RAG.

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class JednoduchaVektorovaDB:
    def __init__(self):
        self.dokumenty = []
        self.vektory = None
        self.vektorizator = TfidfVectorizer()
    
    def pridej_dokumenty(self, docs):
        self.dokumenty.extend(docs)
        self.vektory = self.vektorizator.fit_transform(self.dokumenty)
    
    def vyhledej(self, dotaz, top_k=3):
        dotaz_vektor = self.vektorizator.transform([dotaz])
        podobnosti = cosine_similarity(dotaz_vektor, self.vektory).flatten()
        
        top_indexy = np.argsort(podobnosti)[-top_k:][::-1]
        vysledky = [(self.dokumenty[i], podobnosti[i]) for i in top_indexy]
        
        return vysledky

# Použití
db = JednoduchaVektorovaDB()
db.pridej_dokumenty([
    "Python je programovací jazyk",
    "Strojové učení využívá algoritmy",
    "Vektory reprezentují data numericky"
])

vysledky = db.vyhledej("Co je Python?")
for doc, skore in vysledky:
    print(f"Skóre: {skore:.3f} - {doc}")
```

## 04. Orchestrace více agentů pomocí LangGraph

Vytvořte složité pracovní postupy s více AI agenty pomocí správy stavu a rozhodovací logiky.

```python
from enum import Enum
from typing import Dict, Any

class StavAgenta(Enum):
    ANALYZUJ = "analyzuj"
    VYZKUMEJ = "vyzkumej"
    ODPOVEZ = "odpovez"

class OrkestratorViceAgentu:
    def __init__(self):
        self.agenti = {
            "analyzer": self.analyzuj_dotaz,
            "vyzkumnik": self.vyzkumej_info,
            "odpovidac": self.generuj_odpoved
        }
        self.stav = {}
    
    def analyzuj_dotaz(self, dotaz):
        # Určení typu a složitosti dotazu
        if "?" in dotaz:
            return {"typ": "otazka", "slozitost": "stredni", "dalsi": "vyzkumnik"}
        return {"typ": "prohlaseni", "slozitost": "nizka", "dalsi": "odpovidac"}
    
    def vyzkumej_info(self, kontext):
        # Simulace shromažďování informací
        return {"vyzkumna_data": f"Získané info o: {kontext['dotaz']}", "dalsi": "odpovidac"}
    
    def generuj_odpoved(self, kontext):
        return f"Na základě analýzy: {kontext.get('vyzkumna_data', 'Přímá odpověď')}"
    
    def zpracuj(self, dotaz):
        self.stav = {"dotaz": dotaz}
        
        # Začni s analyzátorem
        analyza = self.agenti["analyzer"](dotaz)
        self.stav.update(analyza)
        
        # Pokračuj podle rozhodnutí
        if analyza["dalsi"] == "vyzkumnik":
            vyzkum = self.agenti["vyzkumnik"](self.stav)
            self.stav.update(vyzkum)
        
        # Generuj finální odpověď
        return self.agenti["odpovidac"](self.stav)

# Použití
orchestrator = OrkestratorViceAgentu()
odpoved = orchestrator.zpracuj("Co je strojové učení?")
print(odpoved)
```

## 05. Pokročilá integrace API pro dynamické odpovědi

Připojte svého asistenta k externím API pro real-time data a rozšířenou funkcionalitu.

```python
import requests
import os
from datetime import datetime

class APIIntegrovanýAsistent:
    def __init__(self):
        self.api_klice = {
            "pocasi": os.getenv("WEATHER_API_KEY"),
            "zpravy": os.getenv("NEWS_API_KEY")
        }
    
    def ziskej_pocasi(self, mesto):
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather"
            params = {
                "q": mesto,
                "appid": self.api_klice["pocasi"],
                "units": "metric"
            }
            odpoved = requests.get(url, params=params)
            data = odpoved.json()
            
            return f"Počasí v {mesto}: {data['weather'][0]['description']}, {data['main']['temp']}°C"
        except Exception as e:
            return f"Data o počasí nejsou dostupná: {str(e)}"
    
    def ziskej_zpravy(self, tema):
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": tema,
                "apiKey": self.api_klice["zpravy"],
                "pageSize": 3,
                "sortBy": "publishedAt"
            }
            odpoved = requests.get(url, params=params)
            clanky = odpoved.json()["articles"]
            
            souhrn_zprav = f"Nejnovější zprávy o {tema}:\n"
            for clanek in clanky:
                souhrn_zprav += f"- {clanek['title']}\n"
            
            return souhrn_zprav
        except Exception as e:
            return f"Data o zprávách nejsou dostupná: {str(e)}"
    
    def zpracuj_pozadavek(self, vstup_uzivatele):
        if "počasí" in vstup_uzivatele.lower():
            # Extrakce názvu města (zjednodušeno)
            mesto = vstup_uzivatele.split("počasí")[-1].strip()
            return self.ziskej_pocasi(mesto or "Praha")
        elif "zprávy" in vstup_uzivatele.lower():
            tema = vstup_uzivatele.split("zprávy o")[-1].strip()
            return self.ziskej_zpravy(tema or "technologie")
        else:
            return "Mohu pomoci s počasím a zprávami. Zkuste se zeptat na počasí nebo zprávy!"

# Použití
asistent = APIIntegrovanýAsistent()
odpoved = asistent.zpracuj_pozadavek("Jaké je počasí?")
print(odpoved)
```

## 06. Monitorování a optimalizace výkonu

Sledujte výkon chatbota a optimalizujte odpovědi pomocí analytiky a zpětnovazebních smyček.

```python
import time
import json
from collections import defaultdict
from datetime import datetime

class MonitorChatbota:
    def __init__(self):
        self.metriky = defaultdict(list)
        self.konverzace = []
    
    def zaznamenej_interakci(self, vstup_uzivatele, odpoved_bota, cas_odpovedi, spokojenost_uzivatele=None):
        interakce = {
            "casova_znacka": datetime.now().isoformat(),
            "vstup_uzivatele": vstup_uzivatele,
            "odpoved_bota": odpoved_bota,
            "cas_odpovedi": cas_odpovedi,
            "spokojenost_uzivatele": spokojenost_uzivatele,
            "delka_vstupu": len(vstup_uzivatele),
            "delka_odpovedi": len(odpoved_bota)
        }
        
        self.konverzace.append(interakce)
        self.metriky["casy_odpovedi"].append(cas_odpovedi)
        self.metriky["delky_vstupu"].append(len(vstup_uzivatele))
        
        if spokojenost_uzivatele:
            self.metriky["skore_spokojenosti"].append(spokojenost_uzivatele)
    
    def ziskej_zprava_vykonu(self):
        if not self.konverzace:
            return "Žádná data nejsou k dispozici"
        
        prumerny_cas_odpovedi = sum(self.metriky["casy_odpovedi"]) / len(self.metriky["casy_odpovedi"])
        prumerna_spokojenost = sum(self.metriky["skore_spokojenosti"]) / len(self.metriky["skore_spokojenosti"]) if self.metriky["skore_spokojenosti"] else 0
        
        return {
            "celkove_interakce": len(self.konverzace),
            "prumerny_cas_odpovedi": round(prumerny_cas_odpovedi, 3),
            "prumerna_spokojenost": round(prumerna_spokojenost, 2),
            "caste_vzory_vstupu": self.analyzuj_vzory()
        }
    
    def analyzuj_vzory(self):
        # Jednoduchá analýza vzorů
        casta_slova = defaultdict(int)
        for konv in self.konverzace:
            slova = konv["vstup_uzivatele"].lower().split()
            for slovo in slova:
                if len(slovo) > 3:  # Filtruj krátká slova
                    casta_slova[slovo] += 1
        
        return dict(sorted(casta_slova.items(), key=lambda x: x[1], reverse=True)[:5])

# Použití s časovacím dekorátorem
def monitoruj_odpoved(monitor, vstup_uzivatele, funkce_odpovedi_bota):
    cas_zacatku = time.time()
    odpoved = funkce_odpovedi_bota(vstup_uzivatele)
    cas_konce = time.time()
    
    monitor.zaznamenej_interakci(vstup_uzivatele, odpoved, cas_konce - cas_zacatku)
    return odpoved

monitor = MonitorChatbota()
```

## 07. Integrace kódu v odpovědích GPT asistenta

Umožněte svému asistentovi spouštět Python kód pro výpočty, zpracování dat a generování dynamického obsahu.

```python
import ast
import operator
import re
from io import StringIO
import sys

class AsistentSVykonavacemKodu:
    def __init__(self):
        self.bezpecne_operatory = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg
        }
    
    def bezpecne_vyhodnoceni(self, vyraz):
        """Bezpečné vyhodnocení matematických výrazů"""
        try:
            uzel = ast.parse(vyraz, mode='eval')
            return self._vyhodnot_uzel(uzel.body)
        except Exception as e:
            return f"Chyba: {str(e)}"
    
    def _vyhodnot_uzel(self, uzel):
        if isinstance(uzel, ast.Constant):
            return uzel.value
        elif isinstance(uzel, ast.BinOp):
            levy = self._vyhodnot_uzel(uzel.left)
            pravy = self._vyhodnot_uzel(uzel.right)
            return self.bezpecne_operatory[type(uzel.op)](levy, pravy)
        elif isinstance(uzel, ast.UnaryOp):
            operand = self._vyhodnot_uzel(uzel.operand)
            return self.bezpecne_operatory[type(uzel.op)](operand)
        else:
            raise ValueError(f"Nepodporovaná operace: {type(uzel)}")
    
    def spust_kod_snippet(self, kod):
        """Spustí bezpečné Python kód snippety"""
        stary_stdout = sys.stdout
        sys.stdout = zachyceny_vystup = StringIO()
        
        try:
            # Povol pouze specifické bezpečné operace
            if any(nebezpecne in kod for nebezpecne in ['import', 'open', 'exec', 'eval']):
                return "Chyba: Detekován nebezpečný kód"
            
            exec(kod)
            vysledek = zachyceny_vystup.getvalue()
            return vysledek if vysledek else "Kód úspěšně spuštěn"
        except Exception as e:
            return f"Chyba při spuštění: {str(e)}"
        finally:
            sys.stdout = stary_stdout
    
    def zpracuj_dotaz_s_kodem(self, dotaz_uzivatele):
        # Detekce požadavků na výpočty
        if re.search(r'vypočítej|spočítej|kolik je \d+', dotaz_uzivatele, re.IGNORECASE):
            # Extrakce matematického výrazu
            math_vzor = r'[\d+\-*/().\s]+'
            shody = re.findall(math_vzor, dotaz_uzivatele)
            if shody:
                vyraz = shody[0].strip()
                vysledek = self.bezpecne_vyhodnoceni(vyraz)
                return f"Výsledek výpočtu: {vyraz} = {vysledek}"
        
        # Detekce požadavků na spuštění kódu
        kod_shoda = re.search(r'```python\n(.*?)```', dotaz_uzivatele, re.DOTALL)
        if kod_shoda:
            kod = kod_shoda.group(1)
            vysledek = self.spust_kod_snippet(kod)
            return f"Výsledek spuštění kódu:\n{vysledek}"
        
        return "Mohu pomoci s výpočty a bezpečným spouštěním kódu. Zkuste mě požádat o výpočet něčeho!"

# Použití
asistent = AsistentSVykonavacemKodu()
odpoved = asistent.zpracuj_dotaz_s_kodem("Vypočítej 15 * 8 + 32")
print(odpoved)

# Příklad spuštění kódu
dotaz_kod = """
Můžeš spustit tento kód?
```python
for i in range(5):
    print(f"Číslo: {i}")
```
"""
print(asistent.zpracuj_dotaz_s_kodem(dotaz_kod))
```

## 08. Design a konfigurace zákaznického asistenta

Navrhněte a nakonfigurujte specializované asistenty přizpůsobené specifickým obchodním potřebám a požadavkům zákazníků.

```python
from dataclasses import dataclass
from typing import List, Dict, Any
import json

@dataclass
class OsobnostAsistenta:
    ton: str  # formální, neformální, přátelský, profesionální
    uroven_expertise: str  # začátečník, pokročilý, expert
    styl_odpovedi: str  # stručný, podrobný, konverzační
    hlas_znacky: str  # korporátní, startupový, osobní

class StavitelZakaznickehoAsistenta:
    def __init__(self):
        self.konfigurace = {}
        self.baze_znalosti = {}
        self.konverzacni_toky = {}
    
    def vytvor_konfiguraci_asistenta(self, nazev_spolecnosti: str, odvetvi: str, 
                                  osobnost: OsobnostAsistenta, 
                                  schopnosti: List[str]):
        konfig = {
            "nazev_spolecnosti": nazev_spolecnosti,
            "odvetvi": odvetvi,
            "osobnost": osobnost.__dict__,
            "schopnosti": schopnosti,
            "systemovy_prompt": self._generuj_systemovy_prompt(nazev_spolecnosti, odvetvi, osobnost),
            "zalozhni_odpovedi": self._generuj_zalohy(osobnost),
            "konverzacni_zacatky": self._generuj_zacatky(odvetvi)
        }
        
        self.konfigurace[nazev_spolecnosti] = konfig
        return konfig
    
    def _generuj_systemovy_prompt(self, spolecnost: str, odvetvi: str, osobnost: OsobnostAsistenta):
        return f"""
        Jsi {osobnost.ton} asistent pro {spolecnost}, společnost v odvětví {odvetvi}.
        
        Tvůj komunikační styl je {osobnost.styl_odpovedi} a tvoje úroveň expertízy je {osobnost.uroven_expertise}.
        Hlas značky: {osobnost.hlas_znacky}.
        
        Pokyny:
        - Vždy udržuj profesionalitu a buď přitom {osobnost.ton}
        - Poskytuj {osobnost.styl_odpovedi} odpovědi
        - Pokud si nejsi jistý, nabídni spojení s lidskou podporou
        - Zaměř se na témata související s {odvetvi}
        - Reprezentuj hodnoty a poslání {spolecnost}
        """
    
    def _generuj_zalohy(self, osobnost: OsobnostAsistenta):
        zalohy = {
            "neznamy_dotaz": f"Nemám konkrétní informace o tom. Spojím tě s naším týmem podpory, který ti lépe pomůže.",
            "technicky_problem": f"Pro technické problémy doporučuji kontaktovat přímo náš technický tým podpory.",
            "stiznost": f"Rozumím tvému problému. Zajistím, aby to dostalo pozornost, kterou si zaslouží, od našeho týmu zákaznických služeb."
        }
        
        if osobnost.ton == "neformální":
            zalohy["neznamy_dotaz"] = "Hmm, tím si nejsem jistý! Najdu někoho, kdo ti s tím pomůže."
        
        return zalohy
    
    def _generuj_zacatky(self, odvetvi: str):
        zacatky = {
            "technologie": [
                "Jak ti dnes mohu pomoci s našimi technickými řešeními?",
                "Hledáš technickou podporu nebo informace o produktech?",
                "Jakou technickou výzvu ti mohu pomoci vyřešit?"
            ],
            "zdravotnictvi": [
                "Jak ti mohu pomoci s tvými zdravotními potřebami?",
                "Máš otázky ohledně našich služeb nebo termínů?",
                "Jsem tu, abych ti pomohl s tvými zdravotními dotazy."
            ],
            "finance": [
                "Jak ti dnes mohu pomoci s tvými finančními potřebami?",
                "Máš otázky ohledně našich finančních služeb?",
                "Jsem tu, abych ti pomohl s bankovními nebo investičními otázkami."
            ]
        }
        
        return zacatky.get(odvetvi, ["Jak ti dnes mohu pomoci?"])
    
    def pridej_bazi_znalosti(self, spolecnost: str, polozky_znalosti: Dict[str, str]):
        if spolecnost not in self.baze_znalosti:
            self.baze_znalosti[spolecnost] = {}
        self.baze_znalosti[spolecnost].update(polozky_znalosti)
    
    def ziskej_odpoved(self, spolecnost: str, dotaz_uzivatele: str):
        if spolecnost not in self.konfigurace:
            return "Asistent není nakonfigurován pro tuto společnost."
        
        konfig = self.konfigurace[spolecnost]
        znalosti = self.baze_znalosti.get(spolecnost, {})
        
        # Jednoduché porovnávání klíčových slov pro demo
        dotaz_lower = dotaz_uzivatele.lower()
        for tema, info in znalosti.items():
            if tema.lower() in dotaz_lower:
                return f"Na základě našich informací: {info}"
        
        # Použij zálohu
        return konfig["zalozhni_odpovedi"]["neznamy_dotaz"]

# Příklad použití
osobnost = OsobnostAsistenta(
    ton="přátelský",
    uroven_expertise="pokročilý", 
    styl_odpovedi="podrobný",
    hlas_znacky="profesionální"
)

stavitel = StavitelZakaznickehoAsistenta()
konfig = stavitel.vytvor_konfiguraci_asistenta(
    nazev_spolecnosti="TechCorp Solutions",
    odvetvi="technologie",
    osobnost=osobnost,
    schopnosti=["product_support", "technical_help", "billing_inquiries"]
)

# Přidej bázi znalostí
stavitel.pridej_bazi_znalosti("TechCorp Solutions", {
    "ceny": "Naše plány začínají na 29 USD měsíčně za základní funkce.",
    "podpora": "Technická podpora je dostupná 24/7 přes chat, e-mail nebo telefon.",
    "funkce": "Nabízíme cloudové úložiště, analýzu dat a API integraci."
})

odpoved = stavitel.ziskej_odpoved("TechCorp Solutions", "Jaké jsou vaše cenové možnosti?")
print(odpoved)
```

## 09. Testování a Optimalizace Zákaznických Asistentů

Implementace komplexních testovacích frameworků a optimalizačních strategií pro produkční chatboty.

```python
import random
import statistics
from datetime import datetime, timedelta
from collections import defaultdict
import matplotlib.pyplot as plt

class ChatbotTester:
    def __init__(self):
        self.test_scenarios = []
        self.test_results = []
        self.performance_metriky = defaultdict(list)
    
    def pridat_testovaci_scenar(self, nazev_scenare: str, testovaci_vstupy: List[str], 
                               ocekavana_klicova_slova: List[str], kriteria_uspesnosti: str):
        scenar = {
            "nazev": nazev_scenare,
            "vstupy": testovaci_vstupy,
            "ocekavana_klicova_slova": ocekavana_klicova_slova,
            "kriteria_uspesnosti": kriteria_uspesnosti,
            "casova_znacka": datetime.now()
        }
        self.test_scenarios.append(scenar)
    
    def spustit_testovaci_sadu(self, chatbot_funkce):
        """Spuštění všech testovacích scénářů proti chatbotu"""
        vysledky = []
        
        for scenar in self.test_scenarios:
            vysledky_scenare = {
                "nazev_scenare": scenar["nazev"],
                "vysledky_testu": [],
                "mira_uspesnosti": 0,
                "prumerna_doba_odpovedi": 0
            }
            
            doby_odpovedi = []
            uspesne_testy = 0
            
            for testovaci_vstup in scenar["vstupy"]:
                cas_zahajeni = datetime.now()
                
                try:
                    odpoved = chatbot_funkce(testovaci_vstup)
                    cas_ukonceni = datetime.now()
                    doba_odpovedi = (cas_ukonceni - cas_zahajeni).total_seconds()
                    doby_odpovedi.append(doba_odpovedi)
                    
                    # Kontrola, zda odpověď obsahuje očekávaná klíčová slova
                    shody_klicovych_slov = sum(1 for klic_slovo in scenar["ocekavana_klicova_slova"] 
                                             if klic_slovo.lower() in odpoved.lower())
                    
                    uspech = shody_klicovych_slov >= len(scenar["ocekavana_klicova_slova"]) * 0.5
                    if uspech:
                        uspesne_testy += 1
                    
                    vysledek_testu = {
                        "vstup": testovaci_vstup,
                        "vystup": odpoved,
                        "doba_odpovedi": doba_odpovedi,
                        "shody_klicovych_slov": shody_klicovych_slov,
                        "uspech": uspech
                    }
                    
                    vysledky_scenare["vysledky_testu"].append(vysledek_testu)
                    
                except Exception as e:
                    vysledky_scenare["vysledky_testu"].append({
                        "vstup": testovaci_vstup,
                        "vystup": f"Chyba: {str(e)}",
                        "doba_odpovedi": 0,
                        "shody_klicovych_slov": 0,
                        "uspech": False
                    })
            
            vysledky_scenare["mira_uspesnosti"] = uspesne_testy / len(scenar["vstupy"])
            vysledky_scenare["prumerna_doba_odpovedi"] = statistics.mean(doby_odpovedi) if doby_odpovedi else 0
            
            vysledky.append(vysledky_scenare)
        
        self.test_results = vysledky
        return vysledky
    
    def vygenerovat_optimalizacni_zprava(self):
        """Generování praktických doporučení pro optimalizaci"""
        if not self.test_results:
            return "Nejsou k dispozici výsledky testů. Nejprve spusťte testy."
        
        zprava = {
            "celkovy_vykon": {},
            "problemove_oblasti": [],
            "doporuceni": []
        }
        
        # Výpočet celkových metrik
        vsechny_miry_uspesnosti = [vysledek["mira_uspesnosti"] for vysledek in self.test_results]
        vsechny_doby_odpovedi = [vysledek["prumerna_doba_odpovedi"] for vysledek in self.test_results]
        
        zprava["celkovy_vykon"] = {
            "prumerna_mira_uspesnosti": statistics.mean(vsechny_miry_uspesnosti),
            "prumerna_doba_odpovedi": statistics.mean(vsechny_doby_odpovedi),
            "pocet_testovanych_scenaru": len(self.test_results)
        }
        
        # Identifikace problémových oblastí
        for vysledek in self.test_results:
            if vysledek["mira_uspesnosti"] < 0.7:
                zprava["problemove_oblasti"].append({
                    "scenar": vysledek["nazev_scenare"],
                    "mira_uspesnosti": vysledek["mira_uspesnosti"],
                    "hlavni_problemy": self._analyzovat_selhani(vysledek["vysledky_testu"])
                })
        
        # Generování doporučení
        if zprava["celkovy_vykon"]["prumerna_mira_uspesnosti"] < 0.8:
            zprava["doporuceni"].append("Zlepšit pokrytí znalostní báze - míra úspěšnosti pod 80%")
        
        if zprava["celkovy_vykon"]["prumerna_doba_odpovedi"] > 2.0:
            zprava["doporuceni"].append("Optimalizovat dobu odezvy - aktuálně nad 2 sekundy")
        
        if len(zprava["problemove_oblasti"]) > 0:
            zprava["doporuceni"].append("Zaměřit se na problémové scénáře s nízkou mírou úspěšnosti")
        
        return zprava
    
    def _analyzovat_selhani(self, vysledky_testu):
        """Analýza běžných vzorců selhání"""
        neuspesne_testy = [t for t in vysledky_testu if not t["uspech"]]
        
        if len(neuspesne_testy) == 0:
            return []
        
        # Jednoduchá analýza vzorců selhání
        caste_problemy = []
        pocet_chyb = sum(1 for t in neuspesne_testy if "Chyba:" in t["vystup"])
        if pocet_chyb > len(neuspesne_testy) * 0.3:
            caste_problemy.append("Vysoká míra chyb - zkontrolovat zpracování výjimek")
        
        kratke_odpovedi = sum(1 for t in neuspesne_testy if len(t["vystup"]) < 50)
        if kratke_odpovedi > len(neuspesne_testy) * 0.5:
            caste_problemy.append("Příliš stručné odpovědi - může chybět dostatečné množství informací")
        
        return caste_problemy

class SbiracZpetneVazby:
    def __init__(self):
        self.data_zpetne_vazby = []
    
    def sbirat_zpetnou_vazbu(self, id_konverzace: str, hodnoceni_uzivatele: int, 
                           text_zpetne_vazby: str, kontext_konverzace: Dict):
        zpetna_vazba = {
            "id_konverzace": id_konverzace,
            "casova_znacka": datetime.now(),
            "hodnoceni": hodnoceni_uzivatele,  # škála 1-5
            "text_zpetne_vazby": text_zpetne_vazby,
            "kontext": kontext_konverzace
        }
        self.data_zpetne_vazby.append(zpetna_vazba)
    
    def analyzovat_trendy_zpetne_vazby(self):
        if not self.data_zpetne_vazby:
            return "Nejsou k dispozici data zpětné vazby"
        
        hodnoceni = [z["hodnoceni"] for z in self.data_zpetne_vazby]
        nedavna_zpetna_vazba = [z for z in self.data_zpetne_vazby 
                               if z["casova_znacka"] > datetime.now() - timedelta(days=7)]
        
        return {
            "prumerne_hodnoceni": statistics.mean(hodnoceni),
            "celkovy_pocet_zpetne_vazby": len(self.data_zpetne_vazby),
            "pocet_nedavne_zpetne_vazby": len(nedavna_zpetna_vazba),
            "rozdeleni_hodnoceni": {i: hodnoceni.count(i) for i in range(1, 6)},
            "caste_stiznosti": self._extrahovat_casta_temata([z["text_zpetne_vazby"] for z in self.data_zpetne_vazby if z["hodnoceni"] < 3])
        }
    
    def _extrahovat_casta_temata(self, negativni_zpetna_vazba):
        # Jednoduchá analýza frekvence klíčových slov
        vsechny_texty = " ".join(negativni_zpetna_vazba).lower()
        slova = vsechny_texty.split()
        frekvence_slov = defaultdict(int)
        
        for slovo in slova:
            if len(slovo) > 4:  # Filtrování krátkých slov
                frekvence_slov[slovo] += 1
        
        return dict(sorted(frekvence_slov.items(), key=lambda x: x[1], reverse=True)[:5])

# Příklad použití
tester = ChatbotTester()

# Přidání testovacích scénářů
tester.pridat_testovaci_scenar(
    nazev_scenare="Informace o produktech",
    testovaci_vstupy=[
        "Jaké produkty nabízíte?",
        "Řekněte mi o vašich službách",
        "Co si u vás mohu koupit?"
    ],
    ocekavana_klicova_slova=["produkt", "služba", "nabízí", "dostupný"],
    kriteria_uspesnosti="Odpověď by měla zmínit produkty nebo služby"
)

tester.pridat_testovaci_scenar(
    nazev_scenare="Požadavky na podporu",
    testovaci_vstupy=[
        "Potřebuji pomoc s mým účtem",
        "Jak resetuji heslo?",
        "Mám technické problémy"
    ],
    ocekavana_klicova_slova=["pomoc", "podpora", "asistence", "kontakt"],
    kriteria_uspesnosti="Odpověď by měla nabídnout pomoc nebo kontakt na podporu"
)

# Mockový chatbot pro testování
def mockovy_chatbot(uzivatelsky_vstup):
    if "produkt" in uzivatelsky_vstup.lower():
        return "Nabízíme různé softwarové produkty a cloudové služby, které pomohou vašemu podnikání."
    elif "pomoc" in uzivatelsky_vstup.lower() or "podpora" in uzivatelsky_vstup.lower():
        return "Mohu vám s tím pomoci! Prosím, kontaktujte náš tým podpory pro technickou asistenci."
    else:
        return "Jsem zde, abych pomohl! Co pro vás mohu udělat?"

# Spuštění testů
vysledky = tester.spustit_testovaci_sadu(mockovy_chatbot)
optimalizacni_zprava = tester.vygenerovat_optimalizacni_zprava()

print("Optimalizační zpráva:")
print(f"Celková míra úspěšnosti: {optimalizacni_zprava['celkovy_vykon']['prumerna_mira_uspesnosti']:.2%}")
print(f"Průměrná doba odezvy: {optimalizacni_zprava['celkovy_vykon']['prumerna_doba_odpovedi']:.2f}s")
print(f"Doporučení: {optimalizacni_zprava['doporuceni']}")
```

## 10. Budování Emoční Inteligence a Digitálních Dvojčat

Vytváření emocionálně vnímavých asistentů, kteří dokážou detekovat, sledovat a reagovat na emoční stavy uživatelů.

```python
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np

@dataclass
class EmocionalniStav:
    valence: float  # -1 (negativní) až 1 (pozitivní)
    vzruseni: float  # 0 (klidný) až 1 (vzrušený)
    dominance: float  # 0 (submisivní) až 1 (dominantní)
    jistota: float  # 0 až 1 (jak si jsme jisti tímto hodnocením)

class DetektorEmoci:
    def __init__(self):
        self.klicova_slova_emoci = {
            'radost': {'šťastný', 'nadšený', 'potěšený', 'spokojený', 'báječný', 'skvělý', 'úžasný', 'nádherný'},
            'smutek': {'smutný', 'depresivní', 'nešťastný', 'zklamaný', 'rozrušený', 'zdrcený', 'hrozný', 'strašný'},
            'hněv': {'naštvaný', 'rozzuřený', 'wściekły', 'podrážděný', 'rozčilený', 'frustrovaný', 'vztek', 'nenávist'},
            'strach': {'vyděšený', 'bojící_se', 'znepokojený', 'úzkostný', 'nervózní', 'obavující_se', 'vylekaný', 'panika'},
            'překvapení': {'překvapený', 'šokovaný', 'ohromený', 'užaslý', 'nečekaný', 'páni'},
            'odpor': {'nechutný', 'odporný', 'hnusný', 'nemocný', 'hrozný', 'ošklivý'}
        }
        
        self.modifikatory_intenzity = {
            'velmi': 1.5, 'extrémně': 2.0, 'opravdu': 1.3, 'docela': 1.2,
            'trochu': 0.7, 'trošku': 0.6, 'mírně': 0.5, 'úplně': 1.8
        }
    
    def detekovat_emoci(self, text: str) -> Dict[str, float]:
        text_maly = text.lower()
        slova = re.findall(r'\b\w+\b', text_maly)
        
        skore_emoci = {emoc: 0.0 for emoc in self.klicova_slova_emoci.keys()}
        
        for i, slovo in enumerate(slova):
            for emoc, klicova_slova in self.klicova_slova_emoci.items():
                if slovo in klicova_slova:
                    zakladni_skore = 1.0
                    
                    # Kontrola modifikátorů intenzity před slovem emoci
                    if i > 0 and slova[i-1] in self.modifikatory_intenzity:
                        zakladni_skore *= self.modifikatory_intenzity[slova[i-1]]
                    
                    # Kontrola negace
                    if i > 0 and slova[i-1] in ['ne', 'nikdy', 'žádný', 'není', 'nebude']:
                        zakladni_skore *= -0.5
                    elif i > 1 and slova[i-2] in ['ne', 'nikdy', 'žádný']:
                        zakladni_skore *= -0.5
                    
                    skore_emoci[emoc] += zakladni_skore
        
        # Normalizace skóre
        max_skore = max(skore_emoci.values()) if max(skore_emoci.values()) > 0 else 1
        return {emoc: skore/max_skore for emoc, skore in skore_emoci.items()}
    
    def emoci_na_stav(self, emoci: Dict[str, float]) -> EmocionalniStav:
        # Mapování emocí na dimenzionální model (valence, vzrušení, dominance)
        valence = (emoci['radost'] - emoci['smutek'] - emoci['hněv'] - emoci['odpor']) / 2
        vzruseni = (emoci['hněv'] + emoci['strach'] + emoci['překvapení'] - emoci['smutek']) / 2
        dominance = (emoci['hněv'] + emoci['radost'] - emoci['strach'] - emoci['smutek']) / 2
        
        # Výpočet jistoty na základě nejsilnější emoce
        jistota = max(emoci.values())
        
        return EmocionalniStav(
            valence=max(-1, min(1, valence)),
            vzruseni=max(0, min(1, vzruseni + 0.5)),
            dominance=max(0, min(1, dominance + 0.5)),
            jistota=jistota
        )

class DigitalniDvojce:
    def __init__(self, id_uzivatele: str):
        self.id_uzivatele = id_uzivatele
        self.emocionalni_historie = []
        self.osobnostni_rysy = {
            'otevřenost': 0.5,
            'svědomitost': 0.5,
            'extraverze': 0.5,
            'přívětivost': 0.5,
            'neuroticismus': 0.5
        }
        self.preference = {}
        self.vzorce_interakce = {
            'preferovana_delka_odpovedi': 'střední',
            'styl_komunikace': 'neutrální',
            'zajmy_temata': []
        }
    
    def aktualizovat_emocionalni_stav(self, novy_stav: EmocionalniStav, kontext: str):
        polozka = {
            'casova_znacka': datetime.now(),
            'stav': novy_stav,
            'kontext': kontext
        }
        self.emocionalni_historie.append(polozka)
        
        # Uchování pouze nedávné historie (posledních 50 interakcí)
        if len(self.emocionalni_historie) > 50:
            self.emocionalni_historie = self.emocionalni_historie[-50:]
        
        # Aktualizace inference osobnostních rysů na základě vzorců
        self._aktualizovat_inference_osobnosti()
    
    def _aktualizovat_inference_osobnosti(self):
        if len(self.emocionalni_historie) < 5:
            return
        
        nedavne_stavy = [polozka['stav'] for polozka in self.emocionalni_historie[-10:]]
        
        # Jednoduchá inference osobnosti
        prumerna_valence = np.mean([stav.valence for stav in nedavne_stavy])
        prumerne_vzruseni = np.mean([stav.vzruseni for stav in nedavne_stavy])
        variabilita_valence = np.var([stav.valence for stav in nedavne_stavy])
        
        # Aktualizace rysů na základě vzorců
        if prumerna_valence > 0.2:
            self.osobnostni_rysy['extraverze'] = min(1.0, self.osobnostni_rysy['extraverze'] + 0.1)
        
        if variabilita_valence > 0.3:
            self.osobnostni_rysy['neuroticismus'] = min(1.0, self.osobnostni_rysy['neuroticismus'] + 0.1)
        
        if prumerne_vzruseni > 0.6:
            self.osobnostni_rysy['otevřenost'] = min(1.0, self.osobnostni_rysy['otevřenost'] + 0.05)
    
    def ziskat_aktualni_emocionalni_trend(self) -> Dict[str, float]:
        if len(self.emocionalni_historie) < 3:
            return {'trend': 'neutrální', 'stabilita': 1.0}
        
        nedavne_valence = [polozka['stav'].valence for polozka in self.emocionalni_historie[-5:]]
        
        if len(nedavne_valence) >= 2:
            trend = 'zlepšující_se' if nedavne_valence[-1] > nedavne_valence[0] else 'zhoršující_se'
            stabilita = 1.0 - np.var(nedavne_valence)
        else:
            trend = 'neutrální'
            stabilita = 1.0
        
        return {'trend': trend, 'stabilita': max(0, stabilita)}
    
    def doporucit_strategii_odpovedi(self) -> Dict[str, str]:
        if not self.emocionalni_historie:
            return {'ton': 'neutrální', 'delka': 'střední', 'pristup': 'informativní'}
        
        aktualni_stav = self.emocionalni_historie[-1]['stav']
        trend = self.ziskat_aktualni_emocionalni_trend()
        
        # Určení strategie odpovědi na základě emočního stavu
        if aktualni_stav.valence < -0.3:  # Uživatel se zdá rozrušený
            return {
                'ton': 'empatický',
                'delka': 'stručný',
                'pristup': 'podporující',
                'navrhy': ['uznat_pocity', 'nabidnout_pomoc', 'byt_trpeliv']
            }
        elif aktualni_stav.valence > 0.3:  # Uživatel se zdá šťastný
            return {
                'ton': 'nadšený',
                'delka': 'střední',
                'pristup': 'poutavý',
                'navrhy': ['pridat_se_k_energii', 'byt_pozitivni', 'povzbudit_dal']
            }
        elif aktualni_stav.vzruseni > 0.7:  # Uživatel se zdá vzrušený/rozrušený
            return {
                'ton': 'klidný',
                'delka': 'krátký',
                'pristup': 'uklidňující',
                'navrhy': ['byt_jasny', 'strukturovat_odpoved', 'vyhnout_se_prehlceni']
            }
        else:  # Neutrální stav
            return {
                'ton': 'přátelský',
                'delka': 'střední',
                'pristup': 'informativní',
                'navrhy': ['byt_napomocny', 'klast_upresňujici_otazky']
            }

class EmpatickýAsistent:
    def __init__(self):
        self.detektor_emoci = DetektorEmoci()
        self.uzivatelska_dvojcata = {}  # id_uzivatele -> DigitalniDvojce
    
    def ziskat_nebo_vytvorit_dvojce(self, id_uzivatele: str) -> DigitalniDvojce:
        if id_uzivatele not in self.uzivatelska_dvojcata:
            self.uzivatelska_dvojcata[id_uzivatele] = DigitalniDvojce(id_uzivatele)
        return self.uzivatelska_dvojcata[id_uzivatele]
    
    def zpracovat_zpravu(self, id_uzivatele: str, zprava: str) -> str:
        # Detekce emocí ve zprávě
        emoci = self.detektor_emoci.detekovat_emoci(zprava)
        emocionalni_stav = self.detektor_emoci.emoci_na_stav(emoci)
        
        # Aktualizace digitálního dvojčete uživatele
        dvojce = self.ziskat_nebo_vytvorit_dvojce(id_uzivatele)
        dvojce.aktualizovat_emocionalni_stav(emocionalni_stav, zprava)
        
        # Získání strategie odpovědi
        strategie = dvojce.doporucit_strategii_odpovedi()
        
        # Generování empatické odpovědi
        return self._generovat_empaticka_odpoved(zprava, emocionalni_stav, strategie)
    
    def _generovat_empaticka_odpoved(self, zprava: str, stav: EmocionalniStav, strategie: Dict) -> str:
        zakladni_odpoved = "Rozumím, že se na mě obracíte s touto záležitostí."
        
        # Přizpůsobení odpovědi na základě emočního stavu a strategie
        if strategie['pristup'] == 'podporující':
            if stav.valence < -0.5:
                zakladni_odpoved = "Cítím, že vás to opravdu trápí, a chci vám pomoci."
            else:
                zakladni_odpoved = "Rozumím, že vás to znepokojuje."
        
        elif strategie['pristup'] == 'poutavý':
            zakladni_odpoved = "To je skvělé slyšet! Jsem nadšený, že vám mohu s tím pomoci."
        
        elif strategie['pristup'] == 'uklidňující':
            zakladni_odpoved = "Pojďme si to vyřešit krok za krokem."
        
        # Přidání vhodného pokračování na základě obsahu zprávy
        if "pomoc" in zprava.lower():
            zakladni_odpoved += " Jakou konkrétní asistenci potřebujete?"
        elif "?" in zprava:
            zakladni_odpoved += " Dovolte mi odpovědět na vaši otázku."
        else:
            zakladni_odpoved += " Jak vám mohu nejlépe pomoci?"
        
        return zakladni_odpoved

# Příklad použití
asistent = EmpatickýAsistent()

# Simulace konverzace s emotivním sledováním
zpravy = [
    "Jsem opravdu frustrovaný tímto softwarem, pořád se zhroutí!",
    "Děkuji za pomoc, to vlastně fungovalo perfektně!",
    "Obávám se termínu, je toho tolik k udělání",
    "Skvělé! To řešení mi ušetřilo spoustu času!"
]

id_uzivatele = "uzivatel_123"
for zprava in zpravy:
    odpoved = asistent.zpracovat_zpravu(id_uzivatele, zprava)
    print(f"Uživatel: {zprava}")
    print(f"Asistent: {odpoved}")
    print(f"Emoční trend: {asistent.uzivatelska_dvojcata[id_uzivatele].ziskat_aktualni_emocionalni_trend()}")
    print("---")
```

## 11. Plánování budoucího vývoje a pokročilé aplikace

Naplánujte si svou roadmapu vývoje AI a prozkoumejte nejmodernější aplikace v konverzační umělé inteligenci.

```python
from enum import Enum
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

class VyvojovafFaze(Enum):
    PROTOTYP = "prototype"
    MVP = "mvp"  
    BETA = "beta"
    PRODUKCE = "production"
    MASSTABING = "scale"

class AISchopnost(Enum):
    ZAKLADNI_CHAT = "basic_chat"
    KONTEXTOVA_PAMET = "context_memory"
    API_INTEGRACE = "api_integration"
    MULTIMODALNI = "multimodal"
    EMOCIONALNI_AI = "emotional_ai"
    ORCHESTRACE_AGENTU = "agent_orchestration"
    VLASTNI_MODELY = "custom_models"
    UCENI_V_REALNEM_CASE = "real_time_learning"

@dataclass
class VyvojovayCil:
    nazev: str
    popis: str
    pozadovane_schopnosti: List[AISchopnost]
    odhadovane_tydny: int
    predpoklady: List[str]
    metriky_uspechu: List[str]

class RoadmapaProjektuAI:
    def __init__(self):
        self.cile = []
        self.aktualni_schopnosti = set()
        self.dokoncene_cile = []
        self.technologicky_stack = {
            'frameworky': [],
            'apis': [],
            'databaze': [],
            'nasazeni': []
        }
    
    def pridat_cil(self, cil: VyvojovayCil):
        self.cile.append(cil)
    
    def nastavit_aktualni_schopnosti(self, schopnosti: List[AISchopnost]):
        self.aktualni_schopnosti = set(schopnosti)
    
    def generovat_vzdelavaci_cestu(self) -> List[Dict]:
        """Generuje personalizovanou vzdělávací cestu na základě současných dovedností a cílů"""
        vzdelavaci_cesta = []
        dostupne_cile = self.cile.copy()
        
        while dostupne_cile:
            # Najdi cíle, které lze zahájit s aktuálními schopnostmi
            pripravene_cile = []
            for cil in dostupne_cile:
                chybejici_schopnosti = set(cil.pozadovane_schopnosti) - self.aktualni_schopnosti
                if len(chybejici_schopnosti) <= 2:  # Lze začít, pokud chybí ≤2 schopnosti
                    pripravene_cile.append((cil, len(chybejici_schopnosti)))
            
            if not pripravene_cile:
                # Žádné cíle nejsou okamžitě dostupné, navrhnout základní učení
                vzdelavaci_cesta.append({
                    'typ': 'zaklad',
                    'doporuceni': 'Nejprve se zaměřte na budování základních schopností',
                    'navrzovane_schopnosti': list(set.union(*[set(c.pozadovane_schopnosti) for c in dostupne_cile]) - self.aktualni_schopnosti)[:3]
                })
                break
            
            # Seřaď podle nejmenšího počtu chybějících schopností, pak podle odhadovaného času
            pripravene_cile.sort(key=lambda x: (x[1], x[0].odhadovane_tydny))
            dalsi_cil = pripravene_cile[0][0]
            
            krok_uceni = {
                'typ': 'cil',
                'cil': dalsi_cil,
                'chybejici_schopnosti': list(set(dalsi_cil.pozadovane_schopnosti) - self.aktualni_schopnosti),
                'vzdelavaci_zdroje': self._ziskat_vzdelavaci_zdroje(dalsi_cil),
                'odhadovane_dokonceni': datetime.now() + timedelta(weeks=dalsi_cil.odhadovane_tydny)
            }
            
            vzdelavaci_cesta.append(krok_uceni)
            dostupne_cile.remove(dalsi_cil)
            self.aktualni_schopnosti.update(dalsi_cil.pozadovane_schopnosti)
        
        return vzdelavaci_cesta
    
    def _ziskat_vzdelavaci_zdroje(self, cil: VyvojovayCil) -> Dict[str, List[str]]:
        """Doporučí vzdělávací zdroje pro specifický cíl"""
        zdroje = {
            'dokumentace': [],
            'tutorialy': [],
            'projekty': [],
            'komunity': []
        }
        
        zdroje_schopnosti = {
            AISchopnost.ZAKLADNI_CHAT: {
                'dokumentace': ['OpenAI API dokumentace', 'LangChain dokumentace'],
                'tutorialy': ['Vytvoř svého prvního chatbota', 'Základy prompt engineeringu'],
                'projekty': ['Jednoduchý Q&A bot', 'FAQ asistent'],
                'komunity': ['OpenAI Community', 'LangChain Discord']
            },
            AISchopnost.KONTEXTOVA_PAMET: {
                'dokumentace': ['Průvodce vektorovými databázemi', 'Embedding modely'],
                'tutorialy': ['Implementace RAG', 'Správa paměti v chatbotech'],
                'projekty': ['Q&A systém pro dokumenty', 'Bot s pamětí konverzace'],
                'komunity': ['Pinecone komunita', 'Weaviate Discord']
            },
            AISchopnost.API_INTEGRACE: {
                'dokumentace': ['Best practices REST API', 'Metody autentifikace'],
                'tutorialy': ['Vzory integrace API', 'Strategie řešení chyb'],
                'projekty': ['Počasový bot', 'Bot pro agregaci zpráv'],
                'komunity': ['Fóra pro vývoj API', 'Stack Overflow']
            },
            AISchopnost.EMOCIONALNI_AI: {
                'dokumentace': ['Knihovny pro analýzu sentimentu', 'API pro detekci emocí'],
                'tutorialy': ['Budování empatické AI', 'Správa stavu uživatele'],
                'projekty': ['Bot pro sledování nálady', 'Terapeutická konverzační AI'],
                'komunity': ['Skupiny AI etiky', 'Fóra psychologie + AI']
            }
        }
        
        for schopnost in cil.pozadovane_schopnosti:
            if schopnost in zdroje_schopnosti:
                for typ_zdroje, polozky in zdroje_schopnosti[schopnost].items():
                    zdroje[typ_zdroje].extend(polozky)
        
        return zdroje
    
    def sledovat_pokrok(self, nazev_cile: str, procento_dokonceni: float, poznamky: str = ""):
        """Sleduje pokrok na konkrétním cíli"""
        for cil in self.cile:
            if cil.nazev == nazev_cile:
                zaznam_pokroku = {
                    'casova_znacka': datetime.now(),
                    'procento_dokonceni': procento_dokonceni,
                    'poznamky': poznamky
                }
                
                if not hasattr(cil, 'historie_pokroku'):
                    cil.historie_pokroku = []
                cil.historie_pokroku.append(zaznam_pokroku)
                
                if procento_dokonceni >= 100:
                    self.dokoncene_cile.append(cil)
                    self.aktualni_schopnosti.update(cil.pozadovane_schopnosti)
                
                break
    
    def generovat_napady_projektu(self, odvetvi: str = None, slozitost: str = "stredni") -> List[Dict]:
        """Generuje nápady na AI projekty na základě aktuálních schopností"""
        sablony_projektu = {
            'zacatecnik': [
                {
                    'nazev': 'Osobní FAQ Bot',
                    'popis': 'Vytvoř bota, který odpovídá na často kladené otázky o tobě nebo tvém businessu',
                    'pozadovane_schopnosti': [AISchopnost.ZAKLADNI_CHAT],
                    'odvetvi': ['obecne', 'osobni', 'maly_business']
                },
                {
                    'nazev': 'Jednoduchý připomínač úkolů',
                    'popis': 'Postav AI asistenta, který pomáhá sledovat a připomínat úkoly',
                    'pozadovane_schopnosti': [AISchopnost.ZAKLADNI_CHAT, AISchopnost.KONTEXTOVA_PAMET],
                    'odvetvi': ['produktivita', 'osobni']
                }
            ],
            'stredni': [
                {
                    'nazev': 'Asistent zákaznické podpory',
                    'popis': 'Inteligentní bot pro zákaznický servis s integrací znalostní báze',
                    'pozadovane_schopnosti': [AISchopnost.ZAKLADNI_CHAT, AISchopnost.KONTEXTOVA_PAMET, AISchopnost.API_INTEGRACE],
                    'odvetvi': ['e-commerce', 'saas', 'sluzby']
                },
                {
                    'nazev': 'Asistent pro tvorbu obsahu',
                    'popis': 'AI, které pomáhá generovat a optimalizovat obsah pro různé platformy',
                    'pozadovane_schopnosti': [AISchopnost.ZAKLADNI_CHAT, AISchopnost.MULTIMODALNI],
                    'odvetvi': ['marketing', 'media', 'vzdelavani']
                },
                {
                    'nazev': 'Společník pro duševní zdraví',
                    'popis': 'Empatická AI poskytující emocionální podporu a sledování wellnessu',
                    'pozadovane_schopnosti': [AISchopnost.EMOCIONALNI_AI, AISchopnost.KONTEXTOVA_PAMET],
                    'odvetvi': ['zdravotnictvi', 'wellness', 'terapie']
                }
            ],
            'pokrocily': [
                {
                    'nazev': 'Multi-agentní výzkumný asistent',
                    'popis': 'Koordinovaní AI agenti, kteří mohou zkoumat, analyzovat a prezentovat složité informace',
                    'pozadovane_schopnosti': [AISchopnost.ORCHESTRACE_AGENTU, AISchopnost.API_INTEGRACE, AISchopnost.KONTEXTOVA_PAMET],
                    'odvetvi': ['vyzkum', 'poradenstvi', 'finance']
                },
                {
                    'nazev': 'Adaptivní výukový tutor',
                    'popis': 'AI tutor, který přizpůsobuje styl výuky na základě pokroku a preferencí studenta',
                    'pozadovane_schopnosti': [AISchopnost.EMOCIONALNI_AI, AISchopnost.UCENI_V_REALNEM_CASE, AISchopnost.MULTIMODALNI],
                    'odvetvi': ['vzdelavani', 'skoleni', 'korporatni']
                }
            ]
        }
        
        vhodne_projekty = []
        for projekt in sablony_projektu.get(slozitost, []):
            pozadovane_schop = set(projekt['pozadovane_schopnosti'])
            if pozadovane_schop.issubset(self.aktualni_schopnosti):
                if not odvetvi or odvetvi in projekt['odvetvi']:
                    vhodne_projekty.append(projekt)
        
        return vhodne_projekty

class TechnologickyStackAI:
    def __init__(self):
        self.doporuceny_stack = {
            'vyvojove_frameworky': {
                'langchain': {
                    'popis': 'Framework pro vývoj LLM aplikací',
                    'nejlepsi_pro': ['rychle_prototypovani', 'kompozice_retezcu', 'orchestrace_agentu'],
                    'krriva_uceni': 'stredni'
                },
                'langgraph': {
                    'popis': 'Stavové multi-aktorové aplikace s LLM',
                    'nejlepsi_pro': ['slozite_workflow', 'multi_agentni_systemy'],
                    'krriva_uceni': 'vysoka'
                },
                'openai_api': {
                    'popis': 'Přímý API přístup k GPT modelům',
                    'nejlepsi_pro': ['jednoduche_integrace', 'optimalizace_nakladu'],
                    'krriva_uceni': 'nizka'
                }
            },
            'vektorove_databaze': {
                'pinecone': {
                    'popis': 'Spravovaná služba vektorové databáze',
                    'nejlepsi_pro': ['produkni_aplikace', 'skalovatnost'],
                    'ceny': 'podle_vyuziti'
                },
                'chroma': {
                    'popis': 'Open-source embedding databáze',
                    'nejlepsi_pro': ['lokalni_vyvoj', 'cost_sensitive_projekty'],
                    'ceny': 'zdarma'
                },
                'weaviate': {
                    'popis': 'Open-source vektorový vyhledávač',
                    'nejlepsi_pro': ['hybridni_vyhledavani', 'slozita_schemata'],
                    'ceny': 'freemium'
                }
            },
            'monitorovaci_nastroje': {
                'langsmith': {
                    'popis': 'LangChain debugging a monitoring platforma',
                    'nejlepsi_pro': ['langchain_aplikace', 'optimalizace_promptu'],
                    'integrace': 'nativni_langchain'
                },
                'weights_biases': {
                    'popis': 'ML sledování experimentů a monitoring',
                    'nejlepsi_pro': ['trenovani_modelu', 'sprava_experimentu'],
                    'integrace': 'siroky_ml_ekosystem'
                }
            },
            'nasazovaci_platformy': {
                'streamlit': {
                    'popis': 'Rychlý vývoj webových aplikací pro ML/AI',
                    'nejlepsi_pro': ['prototypy', 'interni_nastroje', 'dema'],
                    'slozitost': 'nizka'
                },
                'fastapi': {
                    'popis': 'Vysokovýkonný API framework',
                    'nejlepsi_pro': ['produkni_api', 'integracni_sluzby'],
                    'slozitost': 'stredni'
                },
                'cloud_functions': {
                    'popis': 'Serverless nasazení (AWS Lambda, Google Cloud Functions)',
                    'nejlepsi_pro': ['skalovatna_api', 'optimalizace_nakladu'],
                    'slozitost': 'stredni'
                }
            }
        }
    
    def doporucit_stack(self, typ_projektu: str, velikost_tymu: str, rozpocet: str) -> Dict:
        doporuceni = {
            'primarni_framework': '',
            'vektorova_db': '',
            'monitoring': '',
            'nasazeni': '',
            'oduvodneni': []
        }
        
        # Logika doporučení frameworku
        if typ_projektu in ['prototyp', 'mvp'] and velikost_tymu == 'maly':
            doporuceni['primarni_framework'] = 'openai_api'
            doporuceni['oduvodneni'].append('OpenAI API pro rychlý vývoj s malým týmem')
        elif 'multi_agent' in typ_projektu or 'slozity' in typ_projektu:
            doporuceni['primarni_framework'] = 'langgraph'
            doporuceni['oduvodneni'].append('LangGraph pro složité multi-agentní workflow')
        else:
            doporuceni['primarni_framework'] = 'langchain'
            doporuceni['oduvodneni'].append('LangChain pro vyvážené funkcionality a snadné použití')
        
        # Doporučení vektorové DB
        if rozpocet == 'nizky' or typ_projektu == 'prototyp':
            doporuceni['vektorova_db'] = 'chroma'
            doporuceni['oduvodneni'].append('Chroma pro nákladově efektivní lokální vývoj')
        elif 'produkce' in typ_projektu:
            doporuceni['vektorova_db'] = 'pinecone'
            doporuceni['oduvodneni'].append('Pinecone pro produkční škálovatelnost')
        else:
            doporuceni['vektorova_db'] = 'weaviate'
            doporuceni['oduvodneni'].append('Weaviate pro flexibilní hybridní vyhledávání')
        
        # Doporučení monitoringu
        if doporuceni['primarni_framework'] == 'langchain':
            doporuceni['monitoring'] = 'langsmith'
            doporuceni['oduvodneni'].append('LangSmith pro nativní LangChain integraci')
        else:
            doporuceni['monitoring'] = 'weights_biases'
            doporuceni['oduvodneni'].append('Weights & Biases pro komplexní ML monitoring')
        
        # Doporučení nasazení
        if typ_projektu == 'prototyp':
            doporuceni['nasazeni'] = 'streamlit'
            doporuceni['oduvodneni'].append('Streamlit pro rychlé prototypování a dema')
        elif 'api' in typ_projektu or 'produkce' in typ_projektu:
            doporuceni['nasazeni'] = 'fastapi'
            doporuceni['oduvodneni'].append('FastAPI pro produkčně připravená API')
        else:
            doporuceni['nasazeni'] = 'cloud_functions'
            doporuceni['oduvodneni'].append('Serverless pro škálovatelné, nákladově efektivní nasazení')
        
        return doporuceni

# Příklad použití - Vytvoření vývojové roadmapy
def vytvorit_roadmapu_vyvoje_ai():
    roadmapa = RoadmapaProjektuAI()
    
    # Definuj vývojové cíle
    cile = [
        VyvojovayCil(
            nazev="Základní Chatbot MVP",
            popis="Vytvoř funkční chatbot se základními konverzačními schopnostmi",
            pozadovane_schopnosti=[AISchopnost.ZAKLADNI_CHAT],
            odhadovane_tydny=2,
            predpoklady=["Základy Pythonu", "Pochopení API"],
            metriky_uspechu=["Zvládá 80% základních dotazů", "Doba odpovědi < 2s", "Spokojenost uživatelů > 3.5/5"]
        ),
        VyvojovayCil(
            nazev="Asistent s rozšířenou znalostní bází",
            popis="Přidej znalostní bázi dokumentů a kontextovou paměť",
            pozadovane_schopnosti=[AISchopnost.ZAKLADNI_CHAT, AISchopnost.KONTEXTOVA_PAMET],
            odhadovane_tydny=3,
            predpoklady=["Dokončený základní Chatbot MVP"],
            metriky_uspechu=["Přesné odpovědi ze znalostní báze", "Udržuje kontext konverzace", "Zvládá následné otázky"]
        ),
        VyvojovayCil(
            nazev="Integrovaný business asistent",
            popis="Připoj k externím API a business systémům",
            pozadovane_schopnosti=[AISchopnost.ZAKLADNI_CHAT, AISchopnost.KONTEXTOVA_PAMET, AISchopnost.API_INTEGRACE],
            odhadovane_tydny=4,
            predpoklady=["Asistent s rozšířenou znalostní bází", "Znalost vývoje API"],
            metriky_uspechu=["Úspěšně integruje 3+ API", "Zvládá dotazy na real-time data", "Řešení chyb pro API selhání"]
        ),
        VyvojovayCil(
            nazev="Empatický AI společník",
            popis="Postav emocionálně inteligentní asistent se sledováním stavu uživatele",
            pozadovane_schopnosti=[AISchopnost.EMOCIONALNI_AI, AISchopnost.KONTEXTOVA_PAMET],
            odhadovane_tydny=5,
            predpoklady=["Porozumění analýze sentimentu", "Základy psychologie"],
            metriky_uspechu=["Přesně detekuje emoce uživatele", "Přizpůsobuje odpovědi emočnímu stavu", "Zlepšuje zapojení uživatele"]
        )
    ]
    
    for cil in cile:
        roadmapa.pridat_cil(cil)
    
    # Nastav aktuální schopnosti (výchozí bod uživatele)
    roadmapa.nastavit_aktualni_schopnosti([AISchopnost.ZAKLADNI_CHAT])
    
    # Generuj vzdělávací cestu
    vzdelavaci_cesta = roadmapa.generovat_vzdelavaci_cestu()
    
    # Získej doporučení projektů
    projekty = roadmapa.generovat_napady_projektu(slozitost="stredni")
    
    # Získej doporučení technologického stacku
    tech_stack = TechnologickyStackAI()
    doporuceni_stacku = tech_stack.doporucit_stack("mvp", "maly", "stredni")
    
    return {
        'vzdelavaci_cesta': vzdelavaci_cesta,
        'napady_projektu': projekty,
        'tech_stack': doporuceni_stacku,
        'roadmapa': roadmapa
    }

# Generuj komplexní vývojový plán
vyvojovy_plan = vytvorit_roadmapu_vyvoje_ai()

print("🚀 AI Vývojová roadmapa vygenerována!")
print("\n📚 Vzdělávací cesta:")
for i, krok in enumerate(vyvojovy_plan['vzdelavaci_cesta'][:3]):  # Zobraz první 3 kroky
    if krok['typ'] == 'cil':
        print(f"{i+1}. {krok['cil'].nazev} ({krok['cil'].odhadovane_tydny} týdnů)")
        print(f"   Chybějící dovednosti: {', '.join(krok['chybejici_schopnosti'])}")
    else:
        print(f"{i+1}. Budování základů")
        print(f"   Oblasti zaměření: {', '.join(krok['navrzovane_schopnosti'][:3])}")

print("\n💡 Doporučené nápady projektů:")
for projekt in vyvojovy_plan['napady_projektu'][:2]:
    print(f"• {projekt['nazev']}: {projekt['popis']}")

print("\n🛠️ Doporučení technologického stacku:")
stack = vyvojovy_plan['tech_stack']
print(f"• Framework: {stack['primarni_framework']}")
print(f"• Vektorová DB: {stack['vektorova_db']}")
print(f"• Nasazení: {stack['nasazeni']}")

class PlanovacKariery:
    def __init__(self):
        self.karierni_cesty = {
            'ai_inzenyr': {
                'popis': 'Buduj a nasazuj AI systémy v produkčních prostředích',
                'klicove_dovednosti': ['Python', 'ML/DL', 'Cloudové platformy', 'MLOps'],
                'typicky_postup': ['Junior AI Inženýr', 'AI Inženýr', 'Senior AI Inženýr', 'AI Architekt'],
                'platove_rozpeti': '2M - 5M+ Kč ročně',
                'vyhledy_rustu': 'Velmi vysoké'
            },
            'specialista_konverzacni_ai': {
                'popis': 'Specializuj se na chatboty, hlasové asistenty a konverzační rozhraní',
                'klicove_dovednosti': ['NLP', 'Dialogové systémy', 'UX design', 'Psychologie'],
                'typicky_postup': ['Vývojář chatbotů', 'Návrhář konverzací', 'Senior specialista konv. AI', 'Vedoucí AI Experience'],
                'platove_rozpeti': '1.8M - 4.5M+ Kč ročně',
                'vyhledy_rustu': 'Vysoké'
            },
            'ai_product_manager': {
                'popis': 'Veď AI produktovou strategii a vývoj',
                'klicove_dovednosti': ['Produktová strategie', 'Porozumění AI', 'Analýza dat', 'Uživatelský výzkum'],
                'typicky_postup': ['AI PM', 'Senior AI PM', 'Principal PM', 'VP AI produktů'],
                'platove_rozpeti': '2.3M - 6M+ Kč ročně',
                'vyhledy_rustu': 'Velmi vysoké'
            },
            'ai_konzultant': {
                'popis': 'Pomáhej businessům implementovat AI řešení',
                'klicove_dovednosti': ['Business strategie', 'AI technologie', 'Komunikace', 'Projektový management'],
                'typicky_postup': ['AI Konzultant', 'Senior Konzultant', 'Principal', 'Partner/Zakladatel'],
                'platove_rozpeti': '1.5M - 7.5M+ Kč ročně',
                'vyhledy_rustu': 'Vysoké'
            }
        }
    
    def posoudit_vhodnost(self, zajmy: List[str], aktualni_dovednosti: List[str]) -> Dict[str, float]:
        """Posoudí vhodnost pro různé AI kariérní cesty"""
        skore = {}
        
        for nazev_cesty, info_cesty in self.karierni_cesty.items():
            hodnota = 0.0
            
            # Zkontroluj soulad dovedností
            shody_dovednosti = len(set(aktualni_dovednosti) & set(info_cesty['klicove_dovednosti']))
            hodnota += (shody_dovednosti / len(info_cesty['klicove_dovednosti'])) * 0.6
            
            # Zkontroluj soulad zájmů (zjednodušeno)
            klicova_slova_zajmu = {
                'ai_inzenyr': ['technicke', 'kodovani', 'systemy', 'backend'],
                'specialista_konverzacni_ai': ['konverzace', 'uzivatelska_zkusenost', 'jazyk', 'psychologie'],
                'ai_product_manager': ['strategie', 'business', 'vedeni', 'produkt'],
                'ai_konzultant': ['business', 'strategie', 'komunikace', 'rozmanitost']
            }
            
            if nazev_cesty in klicova_slova_zajmu:
                shody_zajmu = len(set(zajmy) & set(klicova_slova_zajmu[nazev_cesty]))
                hodnota += (shody_zajmu / len(klicova_slova_zajmu[nazev_cesty])) * 0.4
            
            skore[nazev_cesty] = min(1.0, hodnota)
        
        return skore

# Příklad použití pro plánování kariéry
planovac_kariery = PlanovacKariery()
uzivatelske_zajmy = ['technicke', 'konverzace', 'uzivatelska_zkusenost']
uzivatelske_dovednosti = ['Python', 'API vývoj', 'základní ML']

karierni_skore = planovac_kariery.posoudit_vhodnost(uzivatelske_zajmy, uzivatelske_dovednosti)
print("\n🎯 Hodnocení kariérní cesty:")
for cesta, skore in sorted(karierni_skore.items(), key=lambda x: x[1], reverse=True):
    info_cesty = planovac_kariery.karierni_cesty[cesta]
    print(f"• {cesta.replace('_', ' ').title()}: {skore:.1%} shoda")
    print(f"  {info_cesty['popis']}")
    print(f"  Plat: {info_cesty['platove_rozpeti']}")
    print()
```