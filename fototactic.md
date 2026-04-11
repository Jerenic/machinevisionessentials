# Fototaktik – Balanced Dataset (8 Candy-Klassen)

## Ziel

- **8 verschiedene Candy-Typen** – jede Klasse soll **gleich oft** im Datensatz vorkommen.
- Zielgröße **ca. 100 Bilder** gesamt.

## Aufteilung (gleiche Klassenhäufigkeit)

100 ist **nicht** durch 8 teilbar (\(100 \div 8 = 12{,}5\)). Streng gleich geht nur mit Vielfachen von 8:

| Gesamtanzahl | Bilder pro Candy |
|--------------|------------------|
| **96**       | **12** × 8       |
| **104**      | **13** × 8       |

**Empfehlung:** Entweder **96 Bilder** (exakt 12 pro Klasse) oder **104 Bilder** (exakt 13 pro Klasse). Wenn die Vorgabe „100“ verbindlich ist, bleibt nur ein **minimaler Ausgleich**:

- **100 Bilder:** z. B. **4 Klassen à 13 Bilder** und **4 Klassen à 12 Bilder** (Differenz nur ±1 pro Klasse).

Dokumentiert im Projekt, welche Klasse 12 bzw. 13 Bilder hat, damit die Verteilung nachvollziehbar bleibt.

## Pro Candy: was die ~12–13 Bilder abdecken sollen

Pro Klasse ungefähr **gleiche** Mischung aus (README + sinnvolle Praxis):

1. **Hintergrund** – mindestens 2–3 verschiedene (Tischfarbe, Unterlage, Stoff, …).
2. **Winkel / Kamera** – leicht von oben, schräg, seitlich; nicht immer dieselbe Höhe.
3. **Bestand** im Sichtfeld – einzelne Stücke **und** mehrere Stücke derselben Sorte.
4. **Licht** – z. B. Tageslicht, warmes Lampenlicht, etwas Schatten / keine harte Überbelichtung auf dem Candy.
5. **Scharf & vollständig** – Candy im Rahmen, nicht dauernd am Rand abgeschnitten; möglichst wenig Bewegungsunschärfe.

So ist **pro Klasse** die gleiche „Vielfalt“, nicht nur die gleiche Stückzahl.

## Ablauf (praktisch)

1. **Klassenliste festlegen** – 8 Namen (wie in Label Studio / `data.yaml` später).
2. **Zielzahl wählen** – 96, 100 oder 104 (siehe Tabelle oben).
3. **Pro Candy nacheinander oder rotierend** fotografieren, bis die Quote pro Klasse erreicht ist (Checkliste oder Zähler im Ordner).
4. Rohbilder **nach Klasse** sortieren oder benennen (`maltesers_001.jpg`, …), damit beim Annotieren nichts verwechselt wird.

## Kurz-Checkliste vor dem Shooting

- [ ] 8 Candy-Typen benannt und bereitgelegt  
- [ ] Ziel: **96 / 100 / 104** und **Bilder pro Klasse** notiert  
- [ ] Mindestens 2 Hintergründe, 2 Lichtsituationen, Einzel- und Mehrfach-Stücke pro Klasse angestrebt  
- [ ] Label-Studio-Klassen = dieselben Namen wie in der Fototaktik  

---

*Hinweis: Technische Fixwerte (Auflösung, exakter Abstand) sind im Lab-README nicht vorgegeben – Smartphone mit üblicher Qualität und gleichbleibend **lesbarer** Darstellung der Candys reicht.*
