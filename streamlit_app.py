import streamlit as st
import pandas as pd
import numpy as np
import PIL.Image
import snowflake.connector
import pydeck as pdk
import pickle
import requests
from urllib.error import URLError


tab1,tab2 = st.tabs(["tab1","tab2"])

with tab1:
# Define the app title and favicon
  st.title('How much can you make from the TastyBytes locations?')
  st.markdown("This tab allows you to make predictions on the price of a listing based on the neighbourhood and room type. The model used is a Random Forest Regressor trained on the Airbnb Singapore listings dataset.")
  st.write('Choose a Truck Brand Name, City, Truck Location and Time Frame to get the predicted sales.')

  with open('xgb_alethea.pkl', 'rb') as file:
        xgb_alethea = pickle.load(file)

  # Load the cleaned and transformed dataset
  df = pd.read_csv('df_alethea1.csv')
  sales = df[['WEEKLY_SALES']] # Extract weekly sales, the target variable

  bn_mapping = { "Cheeky Greek": 0,
                  "Guac n' Roll": 1,
                  "Smoky BBQ": 2,
                  "Peking Truck": 3,
                  "Tasty Tibs": 4,
                  "Better Off Bread": 5,
                  "The Mega Melt": 6,
                  "Le Coin des Crêpes": 7,
                  "The Mac Shack": 8,
                  "Nani's Kitchen": 9,
                  "Plant Palace": 10,
                  "Kitakata Ramen Bar": 11,
                  "Amped Up Franks": 12,
                  "Freezing Point": 13,
                  "Revenge of the Curds": 14 }
  bn_reverse_mapping = {v: k for k, v in bn_mapping.items()}
  bn_labels = list(bn_mapping.keys())

  ct_mapping = { 'San Mateo': 0, 'Seattle': 1, 'New York City': 2, 'Boston': 3, 'Denver':4 }
  ct_reverse_mapping = {v: k for k, v in ct_mapping.items()}
  ct_labels = list(ct_mapping.keys())

  tl_mapping = { {'Veterans Park': 0, 'City of New York': 1, 'Clason Point Park': 2, 'Stanley Bellevue Park': 3, "Hunt's Point Farmers Market": 4, 'Musee Fogg': 5,
                  'Rainey Park': 6, 'Tiffany Street Pier': 7, "Jeff Koons' Balloon Flower Sculpture at WTC": 8, 'Museum Of Modern Art': 9, 'Garland Street Park': 10,
                  'Crestmoor Park': 11, 'Johnson Habitat Park': 12, 'Westcrest Park': 13, 'Fremont Peak Park': 14, 'Christodora House': 15, 'ArchCare at Mary Manning Walsh Home': 16,
                  'West Magnolia Playfield': 17, 'All About Seniors': 18, 'Southwest Auto Park': 19, 'Haven of Care Assisted Living': 20, 'Ellis': 21, 'Garland David T Park': 22,
                  'Ashley': 23, 'Colorado Tennis Association': 24, 'Ryan Play Area': 25, 'Umana Schoolyard': 26, 'Say Yes To Education Teachers College': 27, 'Harlem River Park': 28, 'City Of Kunming Park': 29,
                  'Avalanche Ice Girls': 30, 'Red Light Camera 6th & Lincoln': 31, 'Operation School Bell': 32, 'Denver Art Museum': 33, 'Crating Technologies': 34, 'Frances Wisebart Jacobs Park': 35,
                  'Douglass Fredrick Park': 36, 'Granary Burying Ground': 37, 'Ez Pass': 38, 'Lo Presti Park': 39, 'Franco Bernabe Indio Park': 40, 'Upholstery Outfitters of Seattle': 41, 'Marine Park': 42,
                  'Silver Lake Park': 43, 'Historic New England': 44, 'Maison Blanche': 45, 'Northwest Blind Repair': 46, 'Faneuil Square': 47, 'Felix Potin Chocolatier': 48,
                  'Bell Street Park Boulevard': 49, 'Medcom': 50, 'Washington Park': 51, 'InnovAge Headquarters': 52, 'Eat Harlem': 53, 'Weider Park': 54, 'It Professional': 55, 'Ambassador Candy Store': 56,
                  'Emeritus at West Seattle': 57, 'Army National Guard Recruiting': 58, 'Evergreen Park': 59, 'St John Vianney Theological Seminary': 60, 'Education Commission of the States': 61,
                  'Siemann Educational': 62, 'Sanderson Gulch Irving and Java': 63, "Lower Allston's Christian Herter Park": 64, 'Conley School Play Yard': 65, 'Colorado Rockies Baseball Club': 66,
                  'City Of Cuernavaca Park': 67, 'Father Duffy Square': 68, 'Wyman Multiple Path Iii': 69, 'Dayspring Villa': 70, 'Morrison George Sr Park': 71, 'Eisenhower Mamie Doud Park': 72,
                  'A Place for Mom': 73, 'Cal Anderson Park': 74, 'Woodland Park Off Leash Area': 75, 'Blitz Bowl Football Bowling': 76, 'Steele Street Park': 77, 'American Museum Of Western Art': 78,
                  'Museum Of Contemporary Art Denver': 79, 'Swansea Park': 80, 'East Portal Viewpoint': 81, 'Dudley Town Common': 82, 'Meadowbrook Playfield': 83, 'Spokane Street Bridge': 84,
                  'Allen Park': 85, 'Hungarian Monument': 86, 'Nutbox': 87, 'Molly Brown Summer House': 88, 'Horiuchi Park': 89, 'Museum Of Comic And Cartoon Art': 90, 'McIntyre Organ Repair Service': 91,
                  'Korean War Veterans Memorial Park': 92, 'Wai Wah Market': 93, 'Amasia College': 94, 'Arborway Overpass Path': 95, 'Damrosch Park': 96, 'Riverside Park South': 97,
                  'Gopher Broke Farm': 98, 'Harlem Carmiceria Hispana & Deli': 99, 'Gapstow Bridge': 100, 'Queens Farm Park': 101, 'Spring Manor': 102, 'Evolving Vox': 103, 'TD Garden': 104,
                  'Fourth Street Park': 105, 'Weatherflow Dba Wind Surf': 106, 'Olmsted Green': 107, 'Central Park Tennis Center': 108, 'Uplands Park': 109, "Chef's Cut": 110, 'Me Kwa Mooks Park': 111,
                  'Dunbarton Woods': 112, 'Hirshorn Park': 113, 'Lightrail Plaza': 114, 'Cherry Creek Gun Club': 115, 'Spectrum Retirement Communities': 116, 'Mass Historical Soc': 117,
                  "Soldiers' & Sailors' Monument": 118, 'Conservancy For Historic Battery Park': 119, 'Union Square Metronome': 120, 'Espresso Repair Experts': 121, 'View Ridge Es Playfield': 122,
                  'Bedford Sub Zero Appliance Repair': 123, 'Northlake Park': 124, 'Hing Hay Park': 125, 'Boston University Bridge': 126, 'Amedei': 127, 'Myrtle Edwards Park': 128, 'Pinnacle City Park Hoa': 129,
                  'Friends of Historic Ft Logan': 130, 'Pulaski Park': 131, 'Greenway Park': 132, 'Futura Food': 133, 'Korean War Veterans Memorial': 134, 'Umass Harborwalk': 135, 'Nathan Hale Stadium Complex': 136,
                  'Are East River Science Park': 137, 'Langone Park': 138, 'Atlantic City Boat Ramp': 139, 'Ny Newstand Candy and Grocery': 140, 'American University of Barbados School of Medicine': 141,
                  'Cakes Confidential': 142, 'TeenLife Media': 143, 'Bradner Gardens Park': 144, 'The Jewish Museum': 145, 'Berklee College Of Music': 146, 'Rockledge Street Urban Wild': 147,
                  'Arsenal North': 148, 'Laurelhurst Beach Club': 149, 'Chinatown Fair Family Fun Center': 150, 'Seaport Common': 151, 'Cuny City College': 152, 'Bar And Grill Park': 153,
                  'Seattle Interactive Media Museum': 154, 'Symphony Road Garden': 155, 'Robert H McWilliams Park': 156, 'Museum Of American Humor': 157, 'PasquinelS Landing': 158, 'Joe Moakley Park': 159,
                  'Lower Woodland Park Playfields': 160, 'Place': 161, "Eidem's Custom Upholstery": 162, 'West Central Grounds Maint': 163, 'Merrill Gardens at Queen Anne': 164, 'C & B Candy Store': 165,
                  'Casse Cou': 166, 'Vernam Barbadoes Peninsula': 167, 'Connections': 168, 'Bridge Gear Park': 169, 'Old Harbor Park': 170, 'Moynihan Spray Deck': 171,
                  "Hillel Foundation of Greater Boston B'nai B'rith": 172, 'Grindline': 173, 'Boston University School of Public Health': 174, 'Aquarium New England': 175, 'City Square Park': 176,
                  'Boston University': 177, 'Denison': 178, 'Chinook Beach Park': 179, 'Jack Block Park': 180, 'TrollS Knoll': 181, 'Architectural Heritage Foundation': 182, 'NP Agency Inc': 183, 'Drummers Grove': 184,
                  'Cinderella Tailors': 185, 'Pilsudski Josef Inst of Amer': 186, 'Village Community Boathouse': 187, 'Wrench Bicycle Workshop': 188, 'Greenwood Park': 189, 'Bodyslimdown Garcinialossfat': 190,
                  'Drug Treatment Centers Manhattan': 191, 'Carson Beach': 192, 'Anatolia College': 193, 'Boston History Center And Museum': 194, 'Arctic Viking Experts': 195, 'York Playground': 196, 
                  'United Machine Shop Service': 197, 'Rainier Beach Urban Farm and Wl': 198, 'Share The Care': 199, 'Guan Hua Candy Store': 200, 'Color Factory': 201, "Christie's Sculpture Garden": 202,
                  'Sugarfina': 203, 'Halal Food': 204, 'Matthes Roland': 205, 'Citarella Gourmet Market Upper West Side': 206, 'Beethoven School Play Area': 207, 'African Meeting House': 208,
                  'Mansfield Street Dog Park': 209, 'Darren Sub Zero Repair Master': 210, 'Powell Barnett Park': 211, 'Spada Homes': 212, 'Commonwealth Avenue Outbound': 213, 'Smith Park Pumptrack': 214,
                  'OReilly Way Court': 215, 'Vietnam War Memorial': 216, 'Senior Guidance': 217, 'Knapp': 218, 'Lindsley Henry S Park': 219, 'Magness Arena': 220, 'Mcmeen': 221, 'City Of Takayama Park': 222,
                  'Cardinal Cushing Memorial Park': 223, 'West Street': 224, 'Kendall and Lenox Streets Garden': 225, 'Boston Design Center Plaza': 226, 'Suffolk University Law School': 227, 'Pro Arts Consortium': 228,
                  'University College': 229, 'Denver Drug Rehab': 230, 'Seattle Parks and Recreation': 231, 'University of Washington Virology Research Clinic': 232, 'Forney Museum': 233, 'Doull': 234,
                  'Tzun Tzu Military History Museum': 235, 'Northside Park': 236, 'Ercolini Park': 237, 'Arbor Heights Elder Care': 238, 'Forest Hills Preserve': 239, 'Victory Road Park': 240, 'Schmitz Boulevard': 241,
                  'Kirke Park': 242, 'Suboxone Treatment Clinic': 243, 'Fearless Girl': 244, 'Grant Ranch Village Center': 245, 'Fairview': 246, 'South Slope Dog Run': 247, 'Seward Park': 248, 'Porzio Park': 249,
                  'Union Square Plaza': 250, 'Tent City Courtyards': 251, 'Bayswater Street': 252, 'Prezant Sonia': 253, 'Benjamin Franklin Institute Of Technology': 254, 'Samuel Adams Statue': 255,
                  'General Grant National Memorial': 256, 'High Rated Repair': 257, 'NW Caravan Treasures': 258, 'Presbyterian Retirement Communities Northwest': 259, 'Fuller Theological Seminary': 260,
                  'Harriet Tubman Square': 261, 'Chelsea Creek': 262, 'E & L Seafood': 263, 'Schenck': 264, "See's Candies": 265, 'Harvey Park': 266, 'Laderach': 267, 'Arthur Ashe Stadium': 268,
                  'Dr Blanche Lavizzo Park': 269, 'Florence of Seattle': 270, 'I 90 Interchange': 271, 'Leo M Birmingham Parkway': 272, "Green Lake Pitch n' Putt": 273, 'Homewood Park': 274, 'Miller Triangle': 275,
                  'Leonard P Zakim Bunker Hill Memorial Bridge': 276, 'Rose Fitzgerald Kennedy Greenway': 277, 'Bryant Webster': 278, 'Exceptional Voice': 279, 'Senior Care Authority': 280, 'Millennium Bridge': 281,
                  'Dreamland Wax Museum': 282, 'Simmons College': 283, 'Rogers Park': 284, "Jack O' Lantern Journey": 285, 'Harvard University Harvard Medical School': 286, 'Massport Harborwalk': 287,
                  'South Boston Bark Park': 288, 'Williams Street Iii': 289, 'MassArt Art Museum': 290, 'Carniceria La Rosa': 291, 'Colorado Home Care': 292, 'High Line East': 293, 'Invisible Museum': 294,
                  'Crowley Boats': 295, 'Seattle Tacoma Appliance Repair': 296, 'Burke Gilman Trail': 297, 'ABC Appliance Service': 298, 'Bow Bridge': 299, 'Rider Oasis': 300, 'Drug Rehab and Sober Living Seattle': 301,
                  'The Seattle Great Wheel': 302, 'ARS Appliance Repair Service': 303, 'Columbia Road Day Boulevard': 304, 'Clarendon Street Totlot': 305, "Jim's Cobbler Shoppe": 306, 'Montlake Playfield': 307,
                  'National Oceanic and Atmospheric': 308, 'Met Fresh': 309, 'Carleton Highlands': 310, 'Meril Gardens': 311, 'Halal Cart': 312, 'Emerson College': 313, 'Boston Museum': 314, 'Shi Eurasia': 315,
                  'Pasta Resources': 316, 'McPhilomy Commercial Products': 317, 'Aspen University': 318, 'Thomas J Cuite Park': 319, 'Carson': 320, 'Pulaski Park and Playground': 321,
                  'The Langland House Adult Family Home': 322, 'Refrigerator Repair': 323, 'Hilltop Appliance Repair': 324, 'Japan Premium Beef': 325, 'Columbus Park': 326, 'Malaysia Beef Jerky': 327,
                  'Pinecrest Village Park': 328, 'Montefiore Square': 329, 'Jacques Torres Chocolate': 330, 'Webster Park': 331, 'The Laser Dome': 332, 'American Freestyle Alterations': 333, 'Gorman Park': 334,
                  'John Copley Statue': 335, 'Dennis Street Park': 336, 'Museum Of Afro American History': 337, 'Boston University Sargent Choice Nutrition Center': 338, 'Mt Bowdoin Green': 339,
                  'Martinez Joseph P Park': 340, 'Reclining Liberty': 341, 'Twenty Four Sycamores Park': 342, 'Underground Paranormal Experience': 343, 'E C Hughes Playground': 344, 'Bitter Lake Playfield': 345,
                  'American Cultural Exchange': 346, 'Era Living': 347, 'Jefferson Park': 348, 'Macarthur Park': 349, 'Li Lac Chocolates': 350, 'OU Israel Free Spirit': 351,
                  'Society For the Preservation of New England Antqts': 352, 'Commonwealth Pier': 353, 'Judkins Park P Patch': 354, 'Hyperspace': 355, "Harry & Ida's Meat and Supply": 356,
                  'Home Espresso Repair and Cafe': 357, 'Adam Tailoring & Alterations': 358, 'Spruce Street Mini Park': 359, 'Chinatown Park': 360, "Alioto's Garage": 361, 'Tracey Lactation': 362,
                  'Execupark Incorporated': 363, 'San Mateo County Times': 364, 'SSB Kitchen': 365, 'Owens Electric & Solar': 366, 'VIP Petcare': 367, 'Magic Mountain Playground': 368, 'Health Street': 369,
                  'Paul Revere Mall': 370, 'Blake Estates Urban Wild': 371, 'John Hancock Tower': 372, 'Long Wharf Boat Access': 373, 'Pioneer Square': 374, 'Demonstration Garden': 375, 'Mount Baker Park': 376,
                  'Massachusetts Democratic Party': 377, 'Bremen Street Dog Park': 378, 'Sugar and Plumm': 379, 'Arc Watch Works & Engraving': 380, 'Chocolate Covered Everything': 381,
                  'Simmons College Residence Campus': 382, 'Paul Revere Park': 383, 'The Highlands': 384, 'Ace Hardware': 385, 'Apartments at 225 Catalpa St': 386, 'Macadamian': 387, 'Tina Canada Nail Technician': 388,
                  'Shore Vu Laundromat': 389, 'Faenzi Associates': 390, 'Donna Ornitz': 391, 'Bridgepointe': 392, 'Nordstrom': 393, 'Mark Woods BPIA Insurance': 394, 'KG Fitness Studio': 395, 'West Elm': 396,
                  'Snap Fitness': 397, 'Swarovski': 398, 'King Pang Hairdressing': 399, 'Bay Area Sunrooms': 400, 'Sasi Salon': 401, 'Michaels Stores': 402, 'Apartments at 321 Monte Diablo Ave': 403,
                  'Apartments at 20 Hobart Ave': 404, 'Apartments at 602 Cypress Ave': 405, 'Mason James K Insurance': 406, 'Mcc': 407, 'GreenCal Solar': 408, 'Marco Nascimento Brazilian Jiu Jitsu': 409,
                  'Precision Concrete Cutting': 410, 'Townhouses at 821 Laurel Ave': 411, 'Toi Lynn Wyle MS MFT ERYT': 412, 'Best Western Coyote Point Inn': 413, 'California Bank Trust': 414, 'Reflektion': 415,
                  'Eclipz Hair Designs': 416, 'Park Vanderbilt': 417, 'DP Systems': 418, 'Q Fix PC Services': 419, 'Apartments at 211 E Santa Inez Ave': 420, 'Associated Psychological Services': 421, 'Brust Park': 422,
                  'Chanin News Corporation': 423, 'Kernan Farms': 424, 'Seattle Tower': 425, 'East Queen Anne Playground': 426, 'Matthews Beach': 427, 'Open Water Park': 428, 'Waterfall Garden': 429,
                  'Bar S Playground': 430, 'Bellevue Place': 431, "Expeditors Int'l of Washington": 432, 'Charlestown Navy Yard': 433, 'Fire Alarm House Grounds': 434, 'Nonquit Green': 435, 'Design Museum Boston': 436,
                  'Ideal Health Clinic': 437, 'Hi View Apartments': 438, 'LifeMoves First Step for Families': 439, 'Aikido By The Bay': 440, 'St Timothy School': 441, 'Apartments at 1116 1120 Folkstone Ave': 442,
                  'U S First Federal Credit Union': 443, 'Apartments at 3149 Casa De Campo': 444, 'Reservation Road Park': 445, 'Maximum Performance Hydraulics': 446, 'Bitter Lake Open Space Park': 447,
                  'Conley and Tenean Streets Park': 448, 'My Paper Pros': 449, 'Gourmet Today Publctn': 450, 'Stars Clinic': 451, "Muslim Children's Garden": 452, 'San Mateo DMV Office': 453,
                  'San Bruno Dog Obedience School': 454, 'Barastone': 455, 'First Priority Financial': 456, 'Reposturing The Pain Elimination Method': 457, 'Apartments at 603 E 5th Ave': 458,
                  'East 19th Avenue Apartments': 459, 'Diamond Motors': 460, 'M Street Beach': 461, 'Puddingstone Park': 462, 'Hernandez Schoolyard': 463, 'Machate Circle': 464, 'Unigo': 465,
                  'Judkins Park And Playfield': 466, 'Magnolia Greenbelt': 467, 'Louis Valentino Jr Park & Pier': 468, 'Scott Eaton Supreme Lending': 469, 'Jolene Whitley Hair Design & Replacement': 470,
                  'RePlanet Recycling': 471, 'American Prime Financial': 472, "Bo Jonsson's Foreign Car Service": 473, 'Apex Iron': 474, 'Glass Express': 475, 'Financial Title Company': 476, 'Golden 1 Credit Union': 477,
                  'Esthetique European Skin Care Clinic': 478, 'Y2K Nails': 479, 'Gazelle Developmental School': 480, 'Apartments at 731 N Amphlett Blvd': 481, 'Empowerly': 482, 'Williams-Sonoma': 483,
                  'Brooklyn Bridge Park Pier 5': 484, 'Ryan Playground': 485, 'Hayes Park': 486, 'Pine Street Park': 487, 'Woodhaven': 488, 'Pulmonary Wellness & Rehabilitation Center': 489, 'UCSF Medical Center': 490,
                  'Bologna Chiropractic & Sports Care': 491, 'Apartments at 217 Villa Ter': 492, 'Standard Parking': 493, 'Peninsula Golf & Country Club': 494, 'Thai Art of Massage': 495, 'PattenS Cove': 496,
                  'Lewis Mall': 497, 'American Legion Highway': 498, 'Sustainable Table': 499, 'Lulo Restaurant': 500, 'Cronin Playground': 501, 'Muse Beverly Lmft': 502, 'Release From Within': 503,
                  'Wells Fargo ATM': 504, 'K & W Liquors': 505, 'Jibe Promotional Marketing': 506, 'Meetinghouse Hill Overlook': 507, "Jeffery's Dog & Cat Grooming": 508, 'Come C Interiors': 509,
                  'Apartments at 816 Highland Ave': 510, 'Yok Thai Massage': 511, "Seattle Children's Museum": 512, 'Downer Dog Park': 513, 'San Mateo Auto Works': 514, 'Jazz Museum': 515,
                  'Stony Brook Reservation Iii': 516, 'Eurasian Auto Repair': 517, "DiMenna Children's History Museum": 518, 'Asian Street Eats': 519, 'Leonidas Fresh Belgian Chocolates': 520, 'South Beach': 521,
                  'Park Crest': 522, 'Nathan Hale Playfield': 523, 'Simply Frames & Miner Gallery': 524, 'Aquarist World': 525, "Women's Rights Pioneers Monument": 526, 'Givans Creek Woods': 527,
                  'Our Lady Queen Of Angels': 528, 'National Museum Of Mali': 529, 'Babi Yar Park': 530, 'Eastern Star Masonic Retirement Community': 531, 'Wooden Open': 532, 'Village Greens Park': 533,
                  'Prometheus Statue': 534, 'Chelsea Eats': 535, 'Amersfort Park': 536, 'Coney Island Beach and Boardwalk': 537, 'Underwood Park': 538, 'Kaboom Virtual Reality Arcade': 539,
                  'Troy Chavez Memorial Peace Garden': 540, 'Coney Island Beach': 541, 'Columbus Gourmet Food': 542, 'Core Social Justice Cannabis Museum': 543, 'Elc Playlot': 544, 'Armenian Heritage Park': 545,
                  'Meow Wolf Denver Convergence Station': 546, 'Department of Psychiatry of Columbia University': 547, "Bill's Lobby Stand": 548, 'Maple Grove Park': 549, 'Frederick Douglass Park': 550, 
                  'Farmers Insurance Group': 551, 'Bay Motors': 552, 'Laurelwood Shopping Center': 553, 'Lily European Pedi Mani Spa': 554, 'Star Smog San Mateo': 555, 'Mattahunt School Entrance Plaza': 556,
                  'Association of Independent Colleges & Universities of Mass': 557, 'University Of Denver': 558, 'Ball arena formerly Pepsi Center': 559, 'Salmagundi Club Museum': 560, 'Vesey St Edible': 561,
                  'Vintage Motos Museum': 562, 'The Great Lawn Park': 563, 'Veterans Field': 564, "Sloan's Lake Park": 565, 'The Tiger Lily Kitchen': 566, 'Alcoholics Anonymous': 567,
                  'Penniman Road Play Area': 568, 'Vfw Parkway': 569, 'Jimi Hendrix Statue': 570, 'I S 049 Bertha A Dreyfus': 571, 'Black Top Street Hockey': 572, 'Yue Fung Usa Enterprise Incorporated': 573,
                  'Thornton Creek Natural Area': 574, 'Schmitz Preserve Park': 575, 'Seattle Sub Zero Repair': 576, 'Denver Museum Of Nature & Science': 577, 'Samuels': 578, 'Fishback Park': 579,
                  'Bryant Hill Garden': 580, 'Addiction Recovery Hotline Manhattan': 581, 'Rockaway Beach And Boardwalk': 582, 'New York Career Training & Advancement': 583, 'Tottenville Pool': 584,
                  'Gateway Newstands': 585, 'Daniel Carter Beard Mall': 586, 'Hallack Park': 587, 'Silverman Melvin F Park': 588, 'Balfour': 589, 'Russell Fire Club': 590, 'Central Vacuum Service': 591,
                  'Pacific College of Allied Health': 592, 'CommonGround Golf Course': 593, 'Westerleigh Park': 594, 'City Of Ulaanbaatar Park': 595, 'Cycleton Lowry': 596, 'West 14th Candy Store': 597,
                  'Oldest Manhole Cover In NYC': 598, 'Waldorf Schls Fund': 599, 'Invictus Care': 600, 'Seton Falls Park': 601, 'Lewis Mall Harborpark': 602, 'Wheelock College of Education & Human Development': 603,
                  'InnovAge PACE Denver': 604, "Weiskind's Appliance": 605, 'Huguenot Ponds Park': 606, 'Bayaud Park': 607, 'South Street Courts': 608, 'Lowry Open Space': 609, 'Dailey Park': 610,
                  "University Of Denver Pioneers Men's Lacrosse": 611, 'Parker Terrace': 612, 'Seattle Medical & Rehabilitation Center': 613, 'Muscota Marsh': 614, 'Midtown Catch': 615, 'Macombs Dam Park': 616,
                  'Swansea Neighborhood Park': 617, 'Hampden Heights Park': 618, 'Case Gym': 619, 'Maple Sonoma Streets Community Park': 620, 'Lopresti Park': 621, 'Boston Nature Center Visitor Ctr': 622,
                  "The Vilna Shul Boston's Center for Jewish Culture": 623, 'The ED Clinic': 624, 'F 15 Park': 625, 'Boyden Park': 626, 'Columbia University Medical Center Medical School': 627,
                  'The General Theological Seminary': 628, 'Blood Manor': 629, 'Prentis I Frazier Park': 630, 'O Sullivan Art Museum': 631, 'Dry Gulch Trail': 632, "Martha's Garden": 633, 'Garfield Lake Park': 634,
                  'Sanchez Paco Park': 635, 'Shadow Array': 636, 'Arthur Brisbane Memorial': 637, 'Alford Street Bridge': 638, 'Mgh Institute Of Health Professions': 639, 'Norman B Leventhal Park': 640,
                  'They Shall Walk Museum': 641, 'Marshall Park': 642, 'Lowry': 643, 'Inspir Carnegie Hill': 644, 'Oakland': 645, 'Carla Madison Dog Park': 646, 'Magna Carta Park': 647,
                  'International Neural Renewal': 648, 'Wallace Park': 649, 'Garden Place': 650, 'Staats Circle': 651, 'Sugar Town': 652, 'RTV Exotics': 653, 'Worldwide Adult DayCare': 654,
                  'Travel Center Specialists': 655, 'Rosamond Park': 656, 'Sobriety House': 657, 'Envision Response': 658, 'Ruby Hill': 659, 'Mcclain Thomas Ernest Park': 660, 'Harlem Chocolate Factory': 661,
                  'Gruppo Cioccolato Internazionale': 662, 'L J Campbell Walter': 663, 'Nira Rock': 664, 'Long Lane Meeting House': 665, 'Dreiling Ruth Lucille Park': 666, 'Trinity Park': 667, "Dylan's Candy Bar": 668,
                  'Baker Chocolate Factory': 669, 'Rolling Bridge Park': 670, 'Central Seattle Panel of Consultants Inc': 671, 'Lake City Mini Park': 672, 'Licorice Fern Na On Tc': 673, 'Nespresso': 674,
                  'Peoples Computer Museum': 675, 'Superb Custom Tailors': 676, 'Dacia Woodcliff Community Garden': 677, 'Stony Brook Commons Park': 678, 'Pier Sixty New York City Event and Weddings Venue': 679,
                  'AXA Equitable Center': 680, 'University Park': 681, 'Healing Lifes Pains': 682, 'Noras Woods': 683, 'The Summit At First Hill': 684, "Boston University Women's Council": 685,
                  'Pendleton Miller Playfield': 686, 'UW Medicine Diabetes Institute': 687, 'Woodland Park': 688, 'Key To Amaze': 689, 'Vladeck Park': 690, 'German House': 691, 'Hope Sculpture By Robert Indiana': 692,
                  'Fresh Creek Nature Preserve': 693, 'Puget Creek Natural Area': 694, 'East Madison Street Ferry Dock': 695, 'Bhy Kracke Park': 696, 'Beacon Hill Playground': 697, 'Olympic Hills Es Playfield': 698,
                  'Orchard Street Ravine': 699, 'New Star Fish Market': 700, 'RAIN Inwood Neighborhood Senior Center': 701, 'Halal Street Meat Cart': 702, 'Lt Joseph Petrosino Park': 703, 'Thorndyke Park': 704,
                  'Pratt Park': 705, 'Broadway': 706, 'Cleveland Playfield': 707, "Terry's Thermador Appliance Experts": 708, 'Seattle Appliance Specialist': 709, "Alex's Watch & Clock Repair": 710,
                  'Washington Care Center': 711, 'New Mark Tailor': 712, 'Schatzie the Butcher': 713, 'Eatside': 714, 'Historic Bostonian Properties': 715, 'Dorchester Park': 716, 'The Courtyards at Mountain View': 717,
                  'Community College Of Aurora': 718, 'Westerly Creek Park': 719, 'New York City Hall': 720, 'Ballard Sails': 721, 'Western Avenue Senior Housing': 722,
                  'Center For Autism Rehabilitation & Evaluation': 723, 'Casa Bella Home Care Services': 724, 'Thermador Appliance Mavens': 725, 'Lake Union Yacht Center': 726, 'Emerald Harbor Marine': 727,
                  'Sandel Playground': 728, 'Rubber Chicken Museum': 729, 'Ashburton Place Plaza': 730, 'Copley Square Park': 731, 'Walden Street Community Garden': 732, 'Railroad Avenue': 733,
                  'Minton Stable Garden': 734, 'Washington Park Arboretum': 735, 'Seattle Municipal Tower': 736, 'Jefferson Park Golf Course': 737, 'P S 75 Robert E Peary': 738, 'East River Waterfront Esplanade': 739,
                  'Judge Charles M Stokes Overlk': 740, 'Plymouth Pillars Park': 741, 'Ne 60Th Street Park': 742, 'Twelfth West And W Howe Park': 743, 'Equity Park': 744, "DU's historic Chamberlin Observatory": 745,
                  'Alcohol Drug Rehab Denver': 746, 'The Charles River': 747, 'Comm of Mass Dcr': 748, 'Eastern National': 749, 'New England Aquarium Whale Watch': 750, 'Kennedy Library Harborwalk': 751,
                  'Harambee Park': 752, 'Mass Art Campus': 753, 'Bible James A Park': 754, 'Sabin': 755, 'Inspiration Point Park': 756, 'City Of Karmiel Park': 757, 'Heritage Park': 758, 'Virginia Park': 759,
                  'Metro Park': 760, 'Flourish Supportive Living': 761, 'Hentzel Paul A Park': 762, 'Solstice Park': 763, 'Odyssey Sober Living': 764, 'East Village Meat Market': 765, 'Denison Park': 766,
                  "Gilda's Club New York City": 767, "Raul's Viking Repair Services": 768, 'Sky View Observatory': 769, 'Highland Park': 770, 'Chappetto Square': 771, 'View Ridge Playfield': 772,
                  'Kerry Park Franklin Place': 773, 'Appliance Masters In Seattle': 774, 'Neponset Valley Parkway': 775, 'Kidsport': 776, 'European Senior Care': 777, 'Allen Mall One': 778, 'Ella Bailey Park': 779,
                  'Space Needle': 780, 'Soundview Playfield': 781, 'Indie Fresh': 782, 'Pritchett Island Beach': 783, 'Somerset Street Plaza': 784, 'Oak Square': 785, 'Coppens Square': 786, 'Starfire Education': 787,
                  'Hot Wheels Auto Body': 788, 'Envirotech Toyota Lexus Scion Independent Auto': 789, 'Hair By Andy': 790, 'Bronstein & Associates Insurance Brokers': 791, 'The Goodlife Nutrition Center': 792,
                  'Speedy Auto & Window Glass': 793, 'Metro United Bank': 794, 'Tamarind Financial Planning': 795, 'Salon Kavi': 796, 'Townhouses at 3111 S Delaware St': 797, 'HarborView Park': 798,
                  'Garden of Peace': 799, 'Hennigan Schoolyard': 800, 'Umass Boston': 801, 'Blue Hill Rock': 802, 'Applied Strategies': 803, 'The Peninsula Garage Door': 804, 'Makeover Galore': 805, 'SFOcloud': 806,
                  'Lear': 807, "Ma's Auto Body": 808, 'Clothesline Laundromat': 809, 'Edgar Allan Poe Statue': 810, 'Jamaica Pond Park': 811, 'Fort Warren': 812, 'Kenmore Square': 813,
                  'Massachusetts College of Art and Design': 814, 'Nell Singer Lilac Walk': 815, 'Sea Breeze Fish Market': 816, 'Queen Ann Drive Bridge': 817, 'Baker Park On Crown Hill': 818, 'Laguna Vista': 819,
                  'Allpoint ATM': 820, 'Win Door Service': 821, 'Sturge Presbyterian Church': 822, 'My Pro Auto Glass': 823, 'Total Wine & More': 824, 'Institute on Aging San Mateo County': 825,
                  'Living Word Chruch Of The Deaf Incorporated': 826, 'Aikido By the Bay': 827, 'National Museum Of Hip Hop': 828, "Wang's Chinese Medicine Center": 829, 'James Manley Park': 830,
                  'Heritage Club At Denver Tech Center': 831, "St Lucy's Schl": 832, 'Alpha Beacon Christian School': 833, 'The Betty Mills Company': 834, 'Olsen Auto Body Repair': 835, "Trag's Market": 836,
                  'The Preston at Hillsdale': 837, 'Camp Couture': 838, 'Condos at 1936 Vista Cay': 839, 'Bayhill Heat & Air Inc': 840, 'Crack Corn': 841, 'Als Music Video & Games': 842, 'St Marks Greenbelt': 843,
                  'Tashkent Park': 844, 'Reel Grrls': 845, 'Quadrivium': 846, 'Creekfront Park': 847, 'Chavez Cesar E Park': 848, 'Ford Barney L Park': 849, 'Boston Long Wharf': 850, 'Truce Garden': 851,
                  'Apartments at 50 W 3rd Ave': 852, 'Apartments at 16 Hobart Ave': 853, 'Condos at 6 Seville Way': 854, "Bonardi's Vacuum & Janitorial": 855, 'Hobart Park': 856, 'Topline Automobile': 857,
                  'Bottom Line Billing Service': 858, 'Cinépolis': 859, 'A B Ernst Park': 860, 'Burke Museum Of Natural History And Culture': 861, "Edward's Refrigerator Repair Service": 862, 'Creative Date': 863,
                  'Pinehurst Pocket Park': 864, 'Bradley': 865, 'Wow Interactions': 866, 'Key Markets Main Office': 867, 'La Leena Nails': 868, "Fred's Market": 869, "Mariner's Greens Home Owners Association": 870,
                  'Shall We Dance Tango': 871, 'Apartments at 1006 Tilton Ave': 872, 'Ann Watters PhD': 873, 'Atria Senior Living': 874, 'Apartments at 200 7th Ave': 875, 'Western Graduate School Of Psychology': 876,
                  'Amos Orchids': 877, 'Marble Hill Playground': 878, 'Pralls Island': 879, 'Laviscount Park': 880, 'Warren Gardens Community Garden': 881, 'Apartments at 35 N El Camino Real': 882,
                  'Nurse Family Partnership': 883, 'Roberts': 884, 'Martin Luther King Jr Park': 885, 'Russell Square Park': 886, 'Cheasty Greenspace': 887, 'San Mateo Electronics Inc': 888, 
                  'Southwest Boston Garden Club': 889, 'Edward Everett Hale Statue': 890, 'Forsti Protection Group': 891, 'Babson Cookson Tract': 892, 'Bayview Playground': 893, 'Pot Business Rules For Washington': 894,
                  'Interbay P Patch': 895, "Diane's Alterations and Tailoring": 896, 'Lincoln Square': 897, 'Park Heenan': 898, 'Fort Independence Park': 899, 'Westlake Square': 900, 'One Three Four Elm Apartments': 901,
                  'Paris Fashion Institute': 902, "Eve's Holisitic Animal Studio": 903, 'Dachauer Sheryl M Lep Lmft': 904, 'CoinFlip Bitcoin ATM': 905, 'The Sports Museum': 906, 'Imported Foods': 907,
                  'Visage Esthetiques Skin Care': 908, 'Jacobs Frances Weisbart Park': 909, 'Lincoln': 910, 'Death Museum Store': 911, 'Volunteer Park Conservatory': 912, 'Raoul Wallenberg Forest': 913,
                  'Bay State Road': 914, 'Gas Works Park': 915, 'Ringgold Park': 916, 'Seaport Elite Yacht Charter': 917, 'Thomas C Wales Park': 918, 'East Boston Greenway': 919, 'Atlantic Avenue Plantings': 920,
                  'Tanishas Hood Drag Race': 921, 'Pike Place Market Gum Wall': 922, 'Sugar Zoo': 923, 'Hull Lifesaving Museum': 924, 'US Customs House': 925, 'Campus Convenience': 926, 'Kavod Senior Life': 927,
                  'Emeritus at Roslyn': 928, 'Downtown Childrens Playground': 929, 'Leschi Lake Dell Natural Area': 930, 'Twelfth Avenue South Viewpoint': 931, "Tom's Dog Run": 932, 'Francis Lewis Park': 933,
                  'Parkslope Eatery': 934, 'City Park': 935, 'Steck': 936, 'Shore Road Park': 937, 'Lenox Fish Market': 938, 'Hyosung Park': 939, 'Bryant Neighborhood Playground': 940, 'Court Furniture Rental': 941,
                  'Christopher Columbus Park': 942, 'St CatherineS Park': 943, 'Paerdegat Park': 944, 'Pathfinder': 945, 'Eagleton': 946, 'Montbello Civic Center Park': 947, 'B F Day Playground': 948,
                  'Touch of Elegance Events': 949, 'Fabulous Salon': 950, 'Tai Tung Brake & Muffler Auto Shop': 951, 'Mlc Financial Consulting Services': 952, 'Bay Tree Park': 953, 'Junior Gym': 954,
                  'Bermuda Apartments': 955, 'Crisis Services': 956, 'Metric Tech': 957, 'Alibaba Group': 958, 'Balance Point Strategic Services': 959, 'Hair Designs by Kelly Dinger': 960, 'Bodyrok San Mateo': 961, 
                  'San Mateo Police Department': 962, "Grain D'or": 963, 'Condos at 320 Peninsula Ave': 964, 'Jason Yui Supreme Lending': 965, 'Rose Gold Events': 966, 'Box Ship & More': 967, 'SnapLogic': 968,
                  'Jensen Instrument Company': 969, 'Silicon Valley United Church of Christ': 970, 'Apartments at 419 Rogell Ave': 971, 'Remlo Apartments': 972, 'Angel Nails': 973, 'Townhouses at 106 Franklin Pkwy': 974,
                  'Tat Wong Kung Fu Academy': 975, 'Viking Pavers Construction': 976, 'Belle Isle Coastal Preserve': 977, 'MacKenzie House': 978, 'Sunken Gardens Park': 979, 'Gust': 980, 'Rowe Street Woods': 981,
                  'Park Hill': 982, 'Montclair': 983, 'Cesare Olive Oil & Vinegars': 984, 'Kerry Park': 985, 'Always Perfect Yacht Interiors': 986, 'Golden Way Market': 987, 'Cutillo Park': 988, 'Adams Park': 989,
                  'Cuny Hunter College': 990, 'Hot Soup Cart': 991, 'Central Park Rec Center': 992, 'City Hall Plaza': 993, 'Rink Grounds': 994, 'The Nature Conservancy': 995,
                  'Massachusetts Division of Unemployment Assistance': 996, 'Coors Field': 997, 'Wildlife World Museum': 998, 'Halal Gyro Express': 999}

  def get_TRUCK_BRAND_NAME():
      TRUCK_BRAND_NAME = st.selectbox('Select a truck brand name', bn_mapping)
      return TRUCK_BRAND_NAME
    
  def get_CITY(TRUCK_BRAND_NAME):
    # Only show cities where the selected truck brand works
      cities = df[df['TRUCK_BRAND_NAME'] == bn_mapping[TRUCK_BRAND_NAME]]['CITY'].unique()
      CITY = st.selectbox('Select a city', ct_mapping)
      return CITY

  def get_LOCATION(CITY):
      # Only show truck locations of the selected city
      LOCATION = df[df['CITY'] == bn_mapping[CITY]]['LOCATION'].unique()
      LOCATION = st.selectbox('Select a truck location', tl_mapping)
      return LOCATION  

  # Define the user input fields
  bn_input = get_TRUCK_BRAND_NAME()
  ct_input = get_CITY(bn_input)
  tl_input = get_truckLocation(ct_input)
  
  # Map user inputs to integer encoding
  bn_int = bn_mapping[bn_input]
  ct_int = ct_mapping[ct_input]
  tl_int = tl_mapping[tl_input]
  

with tab2:
  st.title('My Parents New Healthy Diner')

  st.header('Breakfast Favourites')
  st.text('🥣 Omega 3 & Blueberry Oatmeal')
  st.text('🥗 Kale, Spinach & Rocket Smoothie')
  st.text('🐔 Hard-Boiled Free-Range Egg')
  st.text('🥑🍞 Avocado Toast')
  
  # streamlit.header('🍌🥭 Build Your Own Fruit Smoothie 🥝🍇')
  # my_fruit_list = pandas.read_csv("https://uni-lab-files.s3.us-west-2.amazonaws.com/dabw/fruit_macros.txt")
  # my_fruit_list = my_fruit_list.set_index('Fruit')
  
  # # Interactive widget (Multi-select), A pick list to pick the fruit they want to include 
  # fruits_selected = streamlit.multiselect("Pick some fruits:", list(my_fruit_list.index),['Avocado', 'Strawberries'])
  # # Filtering table data
  # fruits_to_show = my_fruit_list.loc[fruits_selected]
  # # Table display
  # streamlit.dataframe(fruits_to_show)
  
  # # Repeatable code block
  # def get_fruityvice_data(this_fruit_choice):
  #   fruityvice_response = requests.get("https://fruityvice.com/api/fruit/"+this_fruit_choice)
  #   fruityvice_normalized = pandas.json_normalize(fruityvice_response.json())
  #   return fruityvice_normalized
  
  # # New Fruityvice API Response
  # streamlit.header("Fruityvice Fruit Advice!")
  # try:
  #   fruit_choice = streamlit.text_input('What fruit would you like information about?')
  #   if not fruit_choice:
  #     streamlit.error("Please select a fruit to get information.")
  #   else:
  #     back_from_function = get_fruityvice_data(fruit_choice)
  #     streamlit.dataframe(back_from_function)
   
  # except URLError as e:
  #   streamlit.error()
  
  # # Fruit Load List
  # streamlit.header("View Our Fruit List – Add Your Favourites!")
  # def get_fruit_load_list():
  #   with my_cnx.cursor() as my_cur:
  #     my_cur.execute("select * from fruit_load_list")
  #     return my_cur.fetchall()
  
  # # Button to load the fruit list
  # if streamlit.button('Get Fruit Load List'):
  #   my_cnx = snowflake.connector.connect(**streamlit.secrets["snowflake"])
  #   my_data_rows = get_fruit_load_list()
  #   my_cnx.close()
  #   streamlit.dataframe(my_data_rows)
  
  # # Allow end user to add fruit to the list
  # def insert_row_snowflake(new_fruit):
  #   with my_cnx.cursor() as my_cur:
  #     my_cur.execute("Insert into fruit_load_list values ('" + new_fruit + "')")
  #     return 'Thanks for adding '+new_fruit
    
  # add_my_fruit = streamlit.text_input('What fruit would you like to add?')
  # if streamlit.button('Add a Fruit to the List'):
  #   my_cnx = snowflake.connector.connect(**streamlit.secrets["snowflake"])
  #   back_from_funcction = insert_row_snowflake(add_my_fruit)
  #   streamlit.text(back_from_function)


