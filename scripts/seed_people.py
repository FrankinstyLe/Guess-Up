"""Seed galleries/people/ with non-STEM majors, so the 'major' lens covers UH.

    python scripts/seed_people.py

The `scientists` gallery is all STEM -- six of its eighteen read Electrical
Engineering -- which makes "you look like someone who majors in..." a demo about
engineering. UH has Liberal Arts, Business, Architecture, Hotel & Restaurant
Management, Social Work, Education, Nursing and Optometry too, and a student who
walks up should be able to land on one of those.

Every entry below is a person whose **undergraduate** field is documented, so
the sentence stays true of the face on screen. Some are deliberately
counter-intuitive and make good table talk: Bruce Lee read philosophy, Yo-Yo Ma
read anthropology, Weird Al read architecture, Harrison Ford read philosophy.

Three are University of Houston alumni, which is worth pointing at when a
prospective student is standing in front of you.

Reuses the fetch/backoff helpers from seed_scientists.py rather than duplicating
them, and writes a `major` field into meta.json for the lens to display.
"""

# System imports
import os
import sys
import json
import time

# Our imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from face_match_kiosk.configs import GALLERIES_DIR
from seed_scientists import fetch_person, REQUEST_DELAY_SECONDS


# (wikipedia title, filename slug, blurb, undergraduate major)
ROSTER = [
    # --- University of Houston alumni
    ('Jim Parsons', 'jim_parsons',
     'UH class of 1996. Twelve years as Sheldon Cooper.',
     'Theatre'),
    ('Brené Brown', 'brene_brown',
     'Three UH degrees. Her TED talk on vulnerability has ~60 million views.',
     'Social Work'),
    ('Dennis Quaid', 'dennis_quaid',
     'Houston born, studied drama at UH before leaving for Hollywood.',
     'Drama'),

    # --- Liberal arts, humanities, social sciences
    ('Barack Obama', 'barack_obama',
     'Majored in political science, then became president.',
     'Political Science'),
    ('Michelle Obama', 'michelle_obama',
     'Sociology at Princeton, then Harvard Law.',
     'Sociology'),
    ('Sonia Sotomayor', 'sonia_sotomayor',
     'Read history, now sits on the Supreme Court.',
     'History'),
    ('Ruth Bader Ginsburg', 'ruth_bader_ginsburg',
     'Government major who rewrote a lot of American law.',
     'Government'),
    ('Thurgood Marshall', 'thurgood_marshall',
     'Literature and philosophy, then Brown v. Board of Education.',
     'Literature & Philosophy'),
    ('Toni Morrison', 'toni_morrison',
     'English major. Nobel Prize in Literature.',
     'English'),
    ('Sandra Cisneros', 'sandra_cisneros',
     'English major from Chicago; wrote The House on Mango Street.',
     'English'),
    ('Gloria Steinem', 'gloria_steinem',
     'Government major who spent a lifetime organising.',
     'Government'),
    ('Dolores Huerta', 'dolores_huerta',
     'Trained as a teacher, then co-founded the United Farm Workers.',
     'Education'),
    ('Martin Luther King Jr.', 'martin_luther_king',
     'Sociology at Morehouse, aged fifteen.',
     'Sociology'),
    ('Amanda Gorman', 'amanda_gorman',
     'Sociology major. Read her poem at a presidential inauguration at 22.',
     'Sociology'),
    ('Chimamanda Ngozi Adichie', 'chimamanda_adichie',
     'Communications and political science; now one of the great novelists.',
     'Communications'),
    ('Bruce Lee', 'bruce_lee',
     'Read philosophy at the University of Washington. Genuinely.',
     'Philosophy'),
    ('Harrison Ford', 'harrison_ford',
     'Philosophy major. Did not graduate. Did fine anyway.',
     'Philosophy'),
    ('Steve Martin', 'steve_martin',
     'Philosophy major; nearly became a professor instead of a comedian.',
     'Philosophy'),
    ('Yo-Yo Ma', 'yo_yo_ma',
     'Anthropology at Harvard, not a music conservatory.',
     'Anthropology'),
    ('Natalie Portman', 'natalie_portman',
     'Psychology at Harvard, with published research.',
     'Psychology'),

    # --- Communications, journalism, media
    ('Oprah Winfrey', 'oprah_winfrey',
     'Speech communications major at Tennessee State.',
     'Communications'),
    ('Christiane Amanpour', 'christiane_amanpour',
     'Journalism major who became the person you see in war zones.',
     'Journalism'),
    ('Anderson Cooper', 'anderson_cooper',
     'Political science major turned correspondent.',
     'Political Science'),

    # --- Performing arts
    ('Meryl Streep', 'meryl_streep',
     'Drama major. Most Oscar nominations in history.',
     'Drama'),
    ('Viola Davis', 'viola_davis',
     'Theatre major; EGOT holder.',
     'Theatre'),
    ("Lupita Nyong'o", 'lupita_nyongo',
     'Film and theatre studies, then an Academy Award.',
     'Film & Theatre'),
    ('John Legend', 'john_legend',
     'English major with an emphasis in African American literature.',
     'English'),
    ('Philip Glass', 'philip_glass',
     'Maths and philosophy at Chicago, at sixteen, before the music.',
     'Mathematics & Philosophy'),

    # --- Architecture and design
    ('Maya Lin', 'maya_lin',
     'Designed the Vietnam Veterans Memorial as an undergraduate.',
     'Architecture'),
    ('Weird Al Yankovic', 'weird_al_yankovic',
     'Architecture major at Cal Poly. Chose accordions instead.',
     'Architecture'),
    ('Zaha Hadid', 'zaha_hadid',
     'Read maths in Beirut before becoming the architect everyone copies.',
     'Mathematics'),
    ('Kehinde Wiley', 'kehinde_wiley',
     'Painting major; painted the official Obama portrait.',
     'Art'),

    # --- Business
    ('Warren Buffett', 'warren_buffett',
     'Business administration major from Nebraska.',
     'Business Administration'),
    ('Howard Schultz', 'howard_schultz',
     'Communications major who turned a coffee shop into Starbucks.',
     'Communications'),

    # --- Food and hospitality
    ('Anthony Bourdain', 'anthony_bourdain',
     'Culinary school after dropping out of Vassar.',
     'Culinary Arts'),
    ('Julia Child', 'julia_child',
     'English major at Smith. Learned to cook much later.',
     'English'),

    # --- Sciences beyond engineering, and a couple of good swerves
    ('Rachel Carson', 'rachel_carson',
     'Biology major whose book started the environmental movement.',
     'Biology'),
    ('Ellen Ochoa', 'ellen_ochoa',
     'Physics major; first Hispanic woman in space.',
     'Physics'),
    ('Mae Jemison', 'mae_jemison',
     'Chemical engineering and African American studies, then space.',
     'Chemical Engineering',),
    ('Ken Jeong', 'ken_jeong',
     'A licensed physician who left medicine for comedy.',
     'Pre-Med / Biology'),
    ('Lisa Kudrow', 'lisa_kudrow',
     'Biology major who researched headaches before Friends.',
     'Biology'),
]


def main():
    directory = os.path.join(GALLERIES_DIR, 'people')
    os.makedirs(directory, exist_ok=True)

    records, failures = [], []

    for title, slug, blurb, major in ROSTER:
        try:
            record, status = fetch_person(title, slug, blurb, directory,
                                          extra={'major': major})
        except Exception as error:
            record, status = None, str(error)

        if record is None:
            print('  [fail] %-26s %s' % (title, status))
            failures.append(title)
        else:
            print('  [%-4s] %-26s %s' % (status, title, major))
            records.append(record)

        time.sleep(REQUEST_DELAY_SECONDS)

    meta_path = os.path.join(directory, 'meta.json')
    with open(meta_path, 'w', encoding='utf-8') as meta_file:
        json.dump(records, meta_file, indent=2, ensure_ascii=False)

    print()
    print('%d portrait(s) in %s' % (len(records), directory))

    if failures:
        print()
        print('No free image found for %d: %s' % (len(failures), ', '.join(failures)))
        print('Living public figures often have no Commons-licensed portrait.')

    fields = sorted({record['major'] for record in records})
    print()
    print('%d distinct major(s): %s' % (len(fields), ', '.join(fields)))
    print()
    print('Next: python scripts/build_gallery.py')
    return 0


if __name__ == '__main__':
    sys.exit(main())
