import streamlit as st
import pandas as pd
import numpy as np
import PIL.Image
import snowflake.connector
import pydeck as pdk
import pickle
import requests
# from lime import lime_tabular
from urllib.error import URLError


tab1,tab2 = st.tabs(["Predicting Daily Sales","tab2"])

with tab1:
  import xgboost as xgb
  
# Define the app title and favicon
  st.title('How much can you make from the TastyBytes locations? 💡')
  st.markdown("Choose a Truck Brand Name, City, Truck Location and Time Frame to get the predicted sales. Using the specific user inputs, the daily sales made by the truck would be predicted.")
  
  # Loading the pickle & dataset
  with open('xgb_alethea.pkl', 'rb') as file:
        xgb_alethea = pickle.load(file)
  df = pd.read_csv('df_aletheaDOW.csv')

  wd_mapping  = { 'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6 }
  wd_reverse_mapping = {v: k for k, v in wd_mapping.items()}
  wd_labels = [wd_reverse_mapping[i] for i in sorted(wd_reverse_mapping.keys())]

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

  tl_mapping = { 'Veterans Park': 0, 'City of New York': 1, 'Clason Point Park': 2, 'Stanley Bellevue Park': 3, "Hunt's Point Farmers Market": 4, 'Musee Fogg': 5,
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
                  'Massachusetts Division of Unemployment Assistance': 996, 'Coors Field': 997, 'Wildlife World Museum': 998, 'Halal Gyro Express': 999, 'Windfall Farms': 1000, 'Ramsay Park': 1001, 
                  'Rocky Mountain Arsenal': 1002, 'Maple School Ravine': 1003, 'Elitch Gardens': 1004, 'Ether Dome': 1005,'Warren Anatomical Museum': 1006, 'Ashland Recreation Center': 1007,
                  'West Roxbury H S Campus': 1008, 'Sargent Street Park': 1009, 'Lil Patch of Heaven': 1010, 'Congress Park': 1011, 'Serenity House Assisted Living Office': 1012, 'Major Mark Park': 1013, 
                  'Benedict Fountain Park': 1014, 'Dewey Square Parks': 1015, 'A MailBoxes & More': 1016, 'Trag\'s Woodwork & Designs': 1017, 'Barcelino Per Donna': 1018, 'Apartments at 2645 S El Camino Real': 1019, 
                  'Centerwood Liquors': 1020, 'Yan\'s Skin Care': 1021, 'Elaine T Valente Open Space': 1022, 'Assited Living Denver': 1023, 'Cherry Creek Greenway': 1024, 'Stevens Triangle': 1025,
                  'Dick Paetzke Creative Directions': 1026, 'Amusement Services': 1027, 'Kilbourne Park': 1028, 'Pioneer Association of the State of Washington': 1029, 'Smith Cove Park': 1030, 'Mary Soo Hoo Park': 1031,
                  'Living Computer Museum': 1032, 'Washington Park Japanese Gdn': 1033, 'Village Place Park': 1034, 'Sonny Lawson Park': 1035, 'University Of Colorado Museum Of Natural History': 1036, 
                  'The Yearling': 1037, 'W H Ferguson Park': 1038, 'Gates Crescent Park': 1039, 'Lucy\'s Whey': 1040, 'Cheltenham': 1041, 'Denver Housing Authority': 1042, 'Weir Gulch Marina Park': 1043, 
                  'Huston Lake Park': 1044, 'Denver Kickball Coalition': 1045, 'Museum Of Anthropology': 1046, 'Carlton Park Foster': 1047, 'Halal Food Cart': 1048, 'Sign Language Center': 1049, 'Mca': 1050, 
                  'EcoGuard Appliance': 1051, 'Waterfront Park': 1052, 'Doors Open Denver': 1053, '82nd Street St Stephen\'s Greenmarket': 1054, 'Ruby Hill Park': 1055, 'Senior Housing Options The Barth Hotel': 1056, 
                  'Washington Park Boathouse': 1057, 'Pelham College': 1058, 'The Boat Rental': 1059, 'Podium Plaza': 1060, 'Counterbalance Park': 1061, 'Spectrum Seniors': 1062, 'Trans World Life Insurance Company': 1063, 
                  'Inpowerfit': 1064, 'Brown William Hair Stylists': 1065, 'Fitness Therapy': 1066, 'Prexion': 1067, 'San Mateo County Community College District Office': 1068, 'Baywood Oaks': 1069, 
                  'Mani Cute Nail Spa': 1070, 'Jiffy Lube': 1071, 'San Mateo Foster City Schl Dist Horrall School': 1072, 'Tian Yuan Taoist Temple': 1073, 'Delta Electric Company': 1074, 'Les Amis Salon et Spa': 1075, 
                  'Quality Towing': 1076, 'Allwin Capital': 1077, 'Boostan Kids': 1078, 'Quincy Street Play Area': 1079, 'Boston University School of Theology': 1080, 'Stonehill Park': 1081, 
                  'Isabella Stewart Gardner Museum': 1082, 'The Paul Revere House': 1083, 'Commercial Point': 1084, 'Mcgann Park': 1085, 'Franklin Park': 1086, 'Msgr Francis A Ryan Park': 1087, 'Newbury Street': 1088, 
                  'Tree Sculpture': 1089, 'Willowwood Rock': 1090, 'Sullivan Square': 1091, 'Geneva Cliffs': 1092, 'Marsh Plaza': 1093, 'University Heights': 1094, 'Old City Hall': 1095, 'Brighton Common': 1096, 
                  'Thea and James M Stoneman Centennial Park': 1097, 'William Ellery Channing Monument': 1098, 'Mount Baker Boulevard': 1099, 'Centennial Park': 1100, 'Good Will Hunting Bench': 1101, 
                  'Permanent Mission of Cuba To the United Nations': 1102, 'Kitchen Masters Specialist': 1103, 'Amira Acne Solutions': 1104, 'Monkey Fist Marine': 1105, 'Rainier Beach Playfield': 1106,
                  'Hunger Intervention Program': 1107, 'Attorney General': 1108, 'Rocky Mountain Arsenal National Wildlife Refuge': 1109, 'Pipers Creek Natural Area': 1110, 'Throgs Neck Park': 1111, 
                  'Fort Point Associates': 1112, B'nan Parcel': 1113, 'Joslin Park': 1114, 'Adventurers Family Entertainment Center': 1115, 'Constitution Plaza': 1116, 'Square Park': 1117, 'Flores Hector M Park': 1118,
                  'West Thames Dog Park': 1119, 'Atlas Autoglass': 1120, 'A 1 Overhead Door Company': 1121, 'Sofia De La Vega': 1122, 'Mother Brook Iii': 1123, 'Codman Square': 1124, 'Thrifty Janitorial': 1125, 
                  'Belltown Cottage Park': 1126, 'Western Motorsports': 1127, 'Northfield Athletic Complex': 1128, 'Mission Main Playlots': 1129, 'Boston University Club': 1130, 'Gerberding Hall': 1131, 
                  'Wood Island Bay Marsh': 1132, 'Brookdale Denver Tech Center': 1133, 'Ashford University': 1134, 'Ny City': 1135, 'Doug\'s Truck & Equipment Repair': 1136, 'Dean Harman': 1137, 
                  'Thrive Integrative Health': 1138, 'C H Robinson Worldwide': 1139, 'Evergreen Towing Service': 1140, 'St James African Methodist Episcopal Zion Church': 1141, 
                  'Apartments at 3828 3830 Colegrove St': 1142, 'Bromwell': 1143, 'Overland Pond Park': 1144, 'Garrison And Union Park': 1145, 'Aztlan Park': 1146, 'Urban Assault Ride by New Belgium': 1147, 
                  'Columbia Rehabilitation and Regenerative Medicine': 1148, 'Intresys TurboCourt': 1149, 'Apartments at 32 N Ellsworth Ave': 1150, 'Carlyle Jewelers': 1151, 'Han\'s Asian Antiques & Arts': 1152, 
                  'TLC Clean & Sober Living Homes': 1153, 'All Pro Plumbing': 1154, 'Nielle Styles': 1155, 'Amino Kit': 1156, 'Technical Information Associates': 1157, 'Lamber Thermador Range Repair Service': 1158,
                   'Metro Tailoring & Alterations': 1159, 'Ridge Custom Upholstery': 1160, 'Barr Dave Plumbing': 1161, 'Yijuan Better Health Clinic': 1162, 'Seattle Public Schools': 1163,
                   'Budget Blinds of West Seattle & Central South': 1164, 'Colorado Film School': 1165, 'Colorado Museum Of Natural History': 1166, 'Fairmont': 1167, 'Kiwanis Ravine Overlook': 1168,
                   'Brighton Police Station Campus': 1169, 'Greenwood Triangle': 1170, 'Sutherland Cooktop Repair': 1171, 'S Lyons': 1172, 'Sorrentino\'s Market': 1173, 'Laredo Avenue Parcel': 1174, 
                   'Juniper Valley Park': 1175, 'Pivot Art Culture': 1176, 'Solo Appliance Repair': 1177, 'Highland Gateway Park': 1178, 'Better Today Addiction Rehab': 1179, 
                   'Independent Growth Home Healthcare Agency': 1180, 'Flushing Greens': 1181, 'Cherry Creek Recreation Area': 1182, 'United States Mint': 1183, 'Mitchell': 1184, 'Bellevue Hill Reservation': 1185,
                   'Cook Park Recreation Center': 1186, 'Independence House Northside': 1187, 'Independence Plaza Parking': 1188, 'East 86Th Ave Trail Greenway': 1189, 'SloanS Lake Park': 1190, 
                   'United Drug Rehab Group Denver': 1191, 'Munroe': 1192, 'Teller': 1193, 'Dickson\'s Fine Brines': 1194, 'Queens Botanical Garden Farmers Market': 1195, 'Pershing Square Plaza': 1196, 
                   'Alexander Hamilton Statue': 1197, 'Traylor': 1198, 'Mestizo Curtis Park': 1199, 'Coast Guard Museum': 1200, 'Slope Park': 1201, 'Sole Repair': 1202, 'Highland Drive Parkway': 1203, 
                   'Roanoke Park Watch Repair': 1204, 'Minetta Triangle': 1205, 'Sandstone Care': 1206, 'Riverfront Park': 1207, 'Flushing Airport': 1208, 'Forest Hills Station Mall': 1209, 'Franklin Park Zoo': 1210, 
                   'Harvard Stadium': 1211, 'West End Museum': 1212, 'Whittier': 1213, 'Marrama': 1214, 'Newlon': 1215, 'China Tea Herbs & Health Products': 1216, 'Spangler Food Court at Harvard Business School': 1217, 
                   'Ronan Park': 1218, 'Boston University Academy': 1219, 'Old North Church Museum': 1220, 'Frieda Garcia Park': 1221, 'Adolescent Psychiatry & Pediatric Psychiatry': 1222, 'Barrett': 1223, 
                   'St PatrickS Neighborhood Park': 1224, 'Little Mystic Access Area': 1225, 'Observatory Park': 1226, 'John F Kennedy Presidential Library And Museum': 1227, 'United States Government': 1228, 
                   'Winthrop Square': 1229, 'Huntington Hemenway Mall': 1230, 'Martini Shell Park Neponset River Reservation': 1231, 'Stanley Ringer Playground': 1232, 'Fenway Park': 1233, 'Stuyvesant Organic': 1234, 
                   'De Witt Clinton Park': 1235, 'Equity Point': 1236, 'Strandway Castle Island': 1237, 'Citgo Sign': 1238, 'Bostonian Society Museum Shop': 1239, 'Lovells Island': 1240, 'Copley Place Plaza': 1241, 
                   'National Consumer Law Center Inc': 1242, 'SoldierS Monument': 1243, 'Flaherty Park': 1244, 'James P Kelleher Rose Garden': 1245, 'African Burial Ground National Monument': 1246, 'Holm': 1247, 
                   'South Bark Dog Park': 1248, 'GameWorks': 1249, 'Tortoise and the Hare Sculpture': 1250, 'Woodcliff': 1251, 'Hallett Fund Academy': 1252, 'A Caring Hand': 1253, 'Ue Enterprises': 1254, 
                   'WolfeS Pond Park': 1255, 'Mott St Senior Center': 1256, 'Aviation & Space Center of the Rockies': 1257, 'Westerly Creek': 1258, 'St Mark\'s Church Greenmarket': 1259, 'illume Electrolysis': 1260, 
                   'Apartments at 739 Highland Ave': 1261, 'Lot Surf': 1262, 'The Big Dig': 1263, 'Boston Nature Center': 1264, 'Belle Isle Marsh Reservation': 1265, 'Fisher Pavillion': 1266, 'Mattahunt Woods Iii': 1267,
                   'Brewer Burroughs Tot Lot': 1268, 'Chinese Historical Society of New England': 1269, 'Williams Fruit Farm': 1270, 'World Trade Center Memorial Foundation': 1271, 'Pferdesteller Park': 1272, 
                   'MGA Crisis Intervention': 1273, 'Denver Phlebotomy School': 1274, 'Berkeley Leashless Dog Park': 1275, 'Sullivan Gateway': 1276, 'Drew University U N Program': 1277, 'Pearly Jones Home': 1278, 
                   'Interbay Golf': 1279, 'Coffee Physics': 1280, 'VR Arcade': 1281, 'Johnson': 1282, 'Burns Dc Park': 1283, 'Cook Judge Joseph E Park': 1284, 'Viking Park': 1285, 'Marston Lake Walking Trail': 1286, 
                   'University Of Denver Pioneers Men\'s Basketball': 1287, 'Decatur West Personal Care': 1288, '5th Avenue': 1289, 'High Line Canal Greenway': 1290, 'Mimi': 1291, 'Sail Park': 1292, 
                   'National American University': 1293, 'Colorado Parks & Wildlife': 1294, 'Apartments at 3120 Los Prados St': 1295, 'Maven Lane Financial Group': 1296, 'Massachusetts Korean War Veterans Memorial': 1297,
                   'Life AcuMed Spa': 1298, 'Bal Aire Mechanical': 1299, 'Eastmoor Park': 1300, 'Harriet Tubman Memorial Statue': 1301, 'Heart 2 Heart 4 Seniors': 1302, 'Scottish Angus Cow and Calf': 1303, 
                   'Stand in Opposition Plaque': 1304, 'Jewels of Highlands House Tour': 1305, 'Carl Schurz Big Dog Run': 1306, 'Hutchinson Theodore A Park': 1307, 'Rosemark At Mayfair Park': 1308, 
                   'City Of Axum Park': 1309, 'Bear Creek Park': 1310, 'Leland Street Herb Garden': 1311, 'School Of The Museum Of Fine Arts At Tufts University': 1312, 'The Humps': 1313, 'Dragonfly Pavilion': 1314, 
                   'Waterway': 1315, 'Harvard Gulch Golf Course': 1316, 'George Washington Bridge Park': 1317, 'Denver School Of Nursing': 1318, 'Marycrest Assisted Living': 1319, 'Mile High Youth Corps': 1320, 
                   'X Gaming': 1321, 'Commons Park': 1322, 'Castro': 1323, 'Molly Brown House Museum': 1324, 'D C Burns Park': 1325, 'Outpatient Women\'s Treatment Services': 1326, 'Yeshiva Toras Chaim': 1327, 
                   'Kirsten Kell Marriage and Family Therapist': 1328, 'Home Instead Senior Care': 1329, 'Deutscher Advisory Group': 1330, 'Apartments at 216 226 6th Ave': 1331, 'Nitsa\'s Barbering': 1332, 
                   'Tech Rocks': 1333, 'Kelly Services': 1334, 'The Children\'s Museum of Denver at Marsico Campus': 1335, 'Cuatro Vientos Four Winds Park': 1336, 'Amesse': 1337, 'Sweet Town': 1338, 
                   'Marion Street Community Center': 1339, 'Shiki Dreams': 1340, 'French & Associates': 1341, 'John Jay Park': 1342, 'Office Depot': 1343, 'The Support Group': 1344, 
                   'Apartments at 120 Peninsula Ave': 1345, 'Great Kills Park Hillcrest': 1346, 'Shivkrupa': 1347, 'Red Auerbach Concourse': 1348, 'Seymour Deborah J Psyd': 1349, 'Archuleta': 1350, 
                   'Bunker Hill Cc Campus Grounds': 1351, 'Shirley Eustis House': 1352, 'Lavietes Pavilion': 1353, 'Columbia Park': 1354, 'Rev Loesch Family Park': 1355, 'Gibson House Museum': 1356, 
                   'Bill Russell Statue': 1357, 'Otis House': 1358, 'Forbes Street Garden': 1359, 'University Of Massachusetts Boston': 1360, 'Stapleton Development Corporation': 1361, 'Pier 10 Mall': 1362, 
                   'Boylston Place': 1363, 'Nordic Museum': 1364, 'Frink Park': 1365, 'City Of Potenza Park': 1366, 'Manley James N Park': 1367, 'Washington Park JJ Byrne Playground': 1368, 
                   'Lavilla Meadows Na On Tc': 1369, 'Fairview Walkway': 1370, 'Miracle Thai': 1371, 'The Dunes Rehab NYC': 1372, 'Hybrid Oak Woods Park': 1373,
                   'Museum At Eldridge Street': 1374, 'Mcphs University': 1375, 'BU Beach': 1376, 'Agganis Arena': 1377, 'Association of Art Museum Directors': 1378, 'Union Bay Natural Area': 1379, 'Steve Herbert': 1380,
                   'Bhangra Empire Classes San Mateo': 1381, 'Js Auto Repair Center': 1382, 'Strategic Awareness Institute': 1383, 'California Catering San Mateo': 1384, 'Hauer Cathy Mft': 1385,
                   'Denise Bloom Interior Design': 1386, 'Elevate Chiropractic Studio': 1387, 'Matt Keck': 1388, 'CenturyLink Tower': 1389, 'University of Colorado At Denver and Health Sciences Center': 1390,
                   'Liberty Day': 1391, 'Thomas ONeill Tunnel': 1392, 'DC Stop Smoking': 1393, 'Mile High Continuing Care': 1394, 'Community Plaza Park': 1395, 'Harvard Gulch West': 1396, 'Lake Of Lakes Park': 1397,
                   'Montage Heights': 1398, 'The Money Museum': 1399, 'East Side Meat Corporation': 1400, 'James Madison Plaza': 1401, 'NYC Department of Parks': 1402, 'Tim\'s DCS Repair Service': 1403,
                   'Denver Roller Derby': 1404, 'Village Shoe Repair': 1405, 'Discovery Park': 1406, 'The Prince From Minneapolis Exhibit': 1407, 'Arrowhead Gardens': 1408, 'Good Shepherd Center': 1409,
                   'Molbak\'s Butterfly Garden': 1410, 'Klondike Gold Rush National Historical Park Seattle Unit': 1411, 'South Boston Maritime Park': 1412, 'Gowanus Waterfront Park': 1413, 'Barnum': 1414,
                   'City Of Chennai Park': 1415, 'Dr Jose Rizal Park': 1416, 'Essential History Expeditions': 1417, 'Westside Church Of Christ': 1418, 'Connect Wireless': 1419, 'Bay Media': 1420,
                   'Alamo Placita Park': 1421, 'Laboratory Institute of Merchandising': 1422, 'Montbello Open Space': 1423, 'Firefighters Memorial Park At Ritz Plaza': 1424, 'Our Heros Food Truck': 1425,
                   'Planeview Park': 1426, 'American Friends of the Bible': 1427, 'Boston Harbor Walk': 1428, 'Castle Square Parks': 1429, 'Prison Ship Martyrs\' Monument': 1430, 'Southwest Corridor Community Farm': 1431,
                   'Arroyo Heights': 1432, 'The Manor on Marston Lake': 1433, 'American College of Greece': 1434, 'John Harvard Mall': 1435, 'Massachusetts State House': 1436, 'Franklin Hill Green': 1437,
                   'Elliot Norton Park': 1438, 'Prince Street Park': 1439, 'USS Growler Submarine': 1440, 'Titus Sparrow Park': 1441, 'Northpoint the Evergreen Northgate': 1442, 'Classic Yacht Association': 1443,
                   'South Ship Canal Trail': 1444, 'Virgil Flaim Park': 1445, 'Seattle Drug Rehab': 1446, 'Ramler Park': 1447, 'Childe Hassam Park': 1448, 'University of Waterloo Manhattan': 1449,
                   'Thrive Center For Healing Traditions': 1450, 'Fairview Park': 1451, 'Tepper Triangle': 1452, 'National Race Masters': 1453, 'William F Passannante Ballfield': 1454, 'Village Pediatrics': 1455,
                   'Revere Plaza': 1456, 'Village Care of New York': 1457, 'Force Basketball': 1458, 'Mcglone': 1459, 'Police Athletic League': 1460, 'City View Park': 1461, 'Wisteria Park': 1462,
                   'Linden Orchard Park': 1463, 'Hornblower Yachts': 1464, 'Asia Society and Museum': 1465, 'African Burial Ground Memorial Office': 1466, 'Owen\'s Candy Stand': 1467, 'Colorado Democratic Party': 1468,
                   'Central Park': 1469, 'Green Valley Ranch East Park': 1470, 'La Salle Deli': 1471, 'Safe Harbor Home Care': 1472, 'Jamaica Plain': 1473, 'Antioch University Seattle': 1474,
                   'Sugar Foods Corporation': 1475, 'Puget Sound': 1476, 'Sobel Court Park': 1477, 'P S 29 Ballfield': 1478, 'Institute of Electrical & Electronic Eng': 1479, 'American Musical And Dramatic Academy': 1480,
                   'Champion Small Engine Repair': 1481, 'Pretzel Time': 1482, 'Soliant Consulting': 1483, 'Lilani Wealth Management': 1484, 'Mind Your Stories & Live Well': 1485, 'Panaderia Guatemalteca Tikal': 1486,
                   'Transportes Ramos': 1487, 'Apartments at 1415 Palm Ave': 1488, 'Hallmark Cards': 1489, 'WinnowPro': 1490, 'Just For Optical': 1491, 'Banana Republic': 1492, 'Bay Meadows East': 1493,
                   'James H Platt Park': 1494, 'Museum Of Computer Technology': 1495, 'Boxer Specialist': 1496, 'Symbolscape Media': 1497, 'Cu Art Museum': 1498, 'San Mateo Auto Care': 1499,
                   'Sharleen\'s Tutoring Service': 1500, 'Macintosh Kenneth M Park': 1501, 'Ravenna Woods': 1502, 'Leventhorp Behavioral': 1503, 'Quincy Coleman Garden': 1504, 'Wartburg College West': 1505,
                   'Overland Park Golf Course': 1506, 'Katie Blacks Garden': 1507, 'Battery Park City Tennis': 1508, 'South Passage Point Park': 1509, 'Alki Point Lighthouse Tours': 1510, 'American Firefighter': 1511,
                   'Transition & Recovery Counseling Center': 1512, 'Rose Garden at Woodland Park Zoo': 1513, 'Macaron Day': 1514, 'Museum Of Holography': 1515, 'The Pearly Gates': 1516, 'Quality Hill Park': 1517,
                   'Parkfield': 1518, 'Boren Park': 1519, 'West Queen Anne Playfield': 1520, 'Northgate Plaza': 1521, 'Terminal 18 Park': 1522, 'KaboomVR': 1523, 'City Square': 1524, 'Washington Huskies Football': 1525,
                   'Balfour Riverfront Park': 1526, 'Churro & Elote Esquite Cart': 1527, 'Western Safety Products Inc': 1528, 'Stonepath Logistics Int\'l Svcs': 1529, 'Divine Hospice & Palliative Care': 1530,
                   'Westwood Park': 1531, 'NYBG Greenmarket': 1532, 'Cottage Hill Senior Apts': 1533, 'The Gardens At St Elizabeth': 1534, 'Palmer': 1535, 'Boat Electric': 1536, 'Martha Washington Park': 1537,
                   'Denver Men\'s Hockey': 1538, 'The Bicycle Repair Shop': 1539, 'Barnum Dog Park': 1540, 'Denver City And County Government Museums': 1541, 'Grant Frontier Park': 1542, 'A Leg Up': 1543,
                   'Martin Luther King Park': 1544, 'Boyd J Langston Park': 1545, 'Children\'s Hospital Colorado KidStreet': 1546, 'Skyline Park': 1547, 'Epoch Assisted Living of Denver': 1548,
                   'Valverde Park': 1549, 'WorldBest Seattle Viking Range Repair': 1550, 'La Familia Recreation Center': 1551, 'Coral Sphere': 1552, 'Espresso Technolgies': 1553, 'Red Square': 1554,
                   'Sandos M L Sam Park': 1555, 'Elenvilla Alterations & Tailoring': 1556, 'Steamworks Espresso Service': 1557, 'Wallingford Steps': 1558, 'Kitchen Repair Specialists': 1559,
                   'Buck Creek Presbyterian Camp': 1560, 'Crown Hill Park': 1561, 'Uysal Alterations': 1562, 'Fremont Canal Park': 1563, 'Central Park Stapleton': 1564, 'Welch Park': 1565,
                   'Lilu Star Indoor Playground': 1566, 'Harvard Gulch Park': 1567, 'E & V Autosales': 1568, 'Sanderson Gulch': 1569, 'Edward Everett Square': 1570, 'Navy Recruiting District Seattle': 1571,
                   'Naval Reserve Armory': 1572, 'JLA Meat Market': 1573, 'Pat Hoffman Friedman Playground Central Park': 1574, 'Ronnybrook Farm Dairy': 1575, 'Motif Investing': 1576, 'Mortgage World': 1577,
                   'Ice Oasis San Mateo': 1578, 'Second Baptist Church of San Mateo': 1579, 'Hari Om Shiri Raadha Krishna Mandir': 1580, 'Sutter Care At Home San Mateo': 1581, 'Franklin Templeton San Mateo': 1582,
                   'Carey School The': 1583, 'Sprint Cellular Service': 1584, 'Honeybee Tattoo': 1585, 'Care Resource Network': 1586, 'Gemenskap Park': 1587, 'The Watch & Jewelry Repair': 1588, 'The ROM Clinic': 1589,
                   'Nob Hill Pizza': 1590, 'CoppS Hill Terrace': 1591, 'Cedar Square': 1592, 'USS Cassin Young': 1593, 'Seaport': 1594, 'iCute iLashes': 1595, 'VR Visage': 1596, 'Lee\'s Comics': 1597,
                   'Candle House': 1598, 'Gateway Park': 1599, 'Primal Pet Foods': 1600, 'Storii Inc': 1601, 'Progressive International Motorcycle Shows': 1602, 'Lead Title Loans': 1603, 'Tom\'s Towing': 1604,
                   'Zutopia Store Hillsdale Mall': 1605, 'State Farm': 1606, 'Andrew Krzeptowski': 1607, 'Metropolitan Apartments': 1608, 'La Hacienda Market': 1609, 'Sears Portrait Studio': 1610,
                   'Shakespeare Garden': 1611, 'Volunteer Parkway': 1612, 'Casa De Dios Puerta Del Cielo': 1613, 'Copps Hill Burying Ground': 1614, 'Shane\'s Barbershop & Shave Parlour': 1615,
                   'Thermador Repair Group Seattle': 1616, 'Housing & Urban Development': 1617, 'Bastion Construction Services': 1618, 'James Hayes Park': 1619, 'Project Mindfull': 1620,
                   'High Amplitude Health Chiropractic and Sports Injury': 1621, 'East Village Cheese': 1622, 'Vinmont Veteran Park': 1623, 'Veterans Memorial Park': 1624, 'Centennial Gardens': 1625,
                   'Fresh&co.': 1626, 'Emeritus at Highline': 1627, 'JP Custom Upolstery and Repair': 1628, 'Queen Anne': 1629, 'Six Diamonds Park': 1630, 'Eckerton Hill Farm': 1631, 'Cranmer Park': 1632,
                   'Verdi Square': 1633, 'Matthews Beach Park': 1634, 'Hutchinson Park': 1635, 'Southwest Corridor Park': 1636, 'Urbani Truffles': 1637, 'Seattle Selfie Museum': 1638, 'Mr. Appliance': 1639,
                   'Seattle Metro Appliance Repair': 1640, 'Luna Park Hoa': 1641, 'Gimbels Traverse': 1642, 'Poseidon Bakery': 1643, 'H & Y Grocery Store': 1644, 'University House Wallingford': 1645,
                   'Project Renewal Clinton Residence': 1646, 'Stanley Sayres Memorial Park': 1647, 'Licton Springs Park': 1648, 'Little Brook Park': 1649, 'Ciancio Park': 1650, 'North Beach Es Playfield': 1651,
                   'Museum Of Communications': 1652, 'Centflor Mfg': 1653, 'Seattle The Top Repair': 1654, 'Max Delivery': 1655, 'Kissena Park': 1656, 'Georgetown Powerplant Museum': 1657,
                   'The Boston Common Frog Pond': 1658, 'Five Points Historic Cultural District': 1659, 'College Point Park': 1660, 'Boston Business School': 1661, 'Herter Park': 1662, 'Athena University': 1663,
                   'Alice Austen Park': 1664, 'Lake City Watch Repair': 1665, 'Ticketmaster': 1666, 'Constitution Beach': 1667, 'Hamilton Viewpoint Park': 1668, 'Einstein Mural by Kobra': 1669,
                   'Lakewood Playfield': 1670, 'Boston Fire Museum': 1671, 'Lincoln Park': 1672, 'Mt Pleasant Play Area': 1673, 'College View': 1674, 'Colorado Heights University': 1675,
                   'Massachusetts Historical Society': 1676, 'Nichols House Museum': 1677, 'Boston Marathon Attack memorial': 1678, 'Skywalk Observatory': 1679, 'Denver Technological Center': 1680,
                   'Filipino American National Historical Society': 1681, 'Curtis Park': 1682, 'Cal Anderson Park and Bobby Morris Playfield': 1683, 'Ballard Swimming Pool': 1684, 'Granada Park Garden': 1685,
                   'Meat Corporation Luis': 1686, 'Best Buy Shellfish': 1687, 'Colfax': 1688, 'Discover the Dinosaurs': 1689, 'Magnolia Park': 1690, 'Dearborn Park': 1691, 'Delridge Playfield': 1692,
                   'Boyer Thermador Appliance Repair': 1693, 'Maple Leaf Reservoir Park': 1694, 'Lakeridge Playfield': 1695, 'Chelsea Street Bridge': 1696, 'USS Constitution': 1697,
                   'Pacific Northwest Museum Of Motorcycling': 1698, 'Marine Corps Recruiting': 1699, 'Madrona Playground': 1700, 'Euston Path Rock': 1701, 'Sanderson Gulch Park & Trail': 1702,
                   'Maple Wood Playfield': 1703, 'Busy Shoes Quality Shoe Repair': 1704, 'Lakewood Moorage': 1705, 'Sweet Gifts Video Cafe': 1706, 'High Line Art': 1707, 'Distortions Monster World': 1708,
                   'Lake City Young People': 1709, 'Logan Field': 1710, 'Seacrest Park': 1711, 'Williamsbridge Oval Park Dog Run': 1712, 'Ward Springs Park': 1713, 'Andover Place': 1714, 'Medal Of Honor Park': 1715,
                   'Brooklyn Academy Of Music': 1716, 'Sub Zero Quality Prime Repair': 1717, 'Concourse Ticket Agency': 1718, 'Allstate Insurance': 1719, 'Kim\'s Nails': 1720, 'A World Driving School': 1721,
                   'San Mateo Farmers\' Market': 1722, 'Mills Peninsula Behavioral Health Services': 1723, 'Zen GE Monogram Repair': 1724, 'Thomas Street Mini Park': 1725, 'Sibby\'s Cupcakery': 1726,
                   'Suffolk University': 1727, 'Arroyos Natural Area': 1728, 'Park Joon': 1729, 'Aquatics Unlimited Instructional Products': 1730, 'The Esplanade': 1731, 'Petals Florist': 1732,
                   'Hat n\' Boots': 1733, 'Harvard Business School Press': 1734, 'Amelia': 1735, 'Permanent Mission of Portugal To the United Nations': 1736, 'United American Bank': 1737,
                   'Flexible Office Space at 999 Baker Way': 1738, 'Fred Tea Shop': 1739, 'Creative Energy': 1740, 'Philbrick Schoolyard': 1741, 'Masonic Hall': 1742, 'Museum Of Fine Arts': 1743,
                   'Walrus Machine': 1744, 'Coronado Pkwy': 1745, 'Altura Park': 1746, 'Chaffee Park': 1747, 'City Market Cafe': 1748, 'Boricua College': 1749, 'McCarren Softball Diamond': 1750,
                   'Columbian': 1751, 'Denver Firefighters Museum': 1752, 'Wallingford Playfield': 1753, 'Royce\' Chocolate': 1754, 'Kelly Outdoor Skating Rink': 1755, 'Rockaway Beach Boardwalk': 1756,
                   'Rehab Helpline': 1757, 'Confectionery': 1758, 'Kings College': 1759, 'Commonwealth Museum': 1760, 'Giant Shoe Museum': 1761, 'Leyland Street Garden': 1762, 'Eunique Associates': 1763,
                   'Helene Madison Pool': 1764, 'Albanese Rudolph Butchr': 1765, 'Slavens': 1766, 'G B A Meats': 1767, 'Emancipation Statue': 1768, 'Smith Cove': 1769, 'Uss Constitution Museum': 1770,
                   'Wallace George M Park': 1771, 'Dolphin Fish Market': 1772, 'Higginson Schoolyard': 1773, 'Sawmill Brook Brook Farm': 1774, 'Nova Southeastern University': 1775,
                   'Kingfisher Natural Area On Tc': 1776, 'CloveS Tail': 1777, 'Smith': 1778, 'Carl Yastrzemski Statue': 1779, 'ChutneyNYC Broadway Bites': 1780, 'Mount Baker Beach': 1781,
                  'Othello Playground': 1782,'Queen Anne Swimming Pool': 1783,'Polar Bear Mechanical': 1784,'Winterdale Arch': 1785,'Dyckman Street Viaduct': 1786,'All the Comfort of Home': 1787,
                  'Cheese Unlimited': 1788,'Dcpa Sculpture Park': 1789,'Athmar Park': 1790,'National Addiction Rehabs Denver': 1791,'Columbia Tower Club': 1792,'Nantes Park': 1793,'College Street Ravine': 1794,
                  'Grady KitchenAid Repair Master': 1795,'Seattle Chinese Garden': 1796,'Admiral Farragut': 1797,'Edison': 1798,'Pagoda Park': 1799,'Boston African American National Historic Site': 1800,
                  'Malibu Beach': 1801,'American Civil Liberties Union': 1802,'Merrill Gardens at First Hill': 1803,'Daniels and Fisher Tower': 1804,'Hillside Calumet': 1805,'Conservatory Green': 1806,
                  'Forest Hills Greenspace': 1807,'Lueck Open Space': 1808,'Pino\'s Prime Meat Market': 1809,'Boston College Brighton Campus': 1810,'RosemaryS Playground': 1811,'International Process Svc': 1812,
                  'Masjid ul Haqq': 1813,'T W Automotive': 1814,'Cornish & Carey': 1815,'Tufts University Boston Campus': 1816,'Neurolink Chiropractic': 1817,'GE Monogram Appliance Repair Service': 1818,
                  'Reggie Wong Memorial Park': 1819,'West Harlem Piers': 1820,'Change': 1821,'The Sacred Church Of The Assemblies Of God': 1822,'Apartments at 245 Grand Blvd': 1823,
                  'Peninsula Custom Framing Company': 1824,'Autokraft': 1825,'Essex Property Trust': 1826,'Dino Motors': 1827,'Wooden Window': 1828,'Apartments at 231 Villa Ter': 1829,
                  'Faicco\'s Italian Specialties': 1830, 'ParrFit': 1831, 'National Ballpark Museum': 1832,'Pilates Instructor Training & PMA Certification': 1833,'Nine Rubies Knitting': 1834,
                  'Kathy McConnell': 1835,'Valentine Nails': 1836,'Fluff & Puff Dog Wash': 1837,'Swansons Shoe Repair': 1838,'Thomas Street Gardens': 1839,'Boston Tea Party Ship And Museum': 1840,
                  'Don Armeni Boat Ramp': 1841,'Brighton Hs Hillside': 1842,'Rainsford Island': 1843,'Camp Meigs': 1844,'Marvins Garden': 1845,'VNA Hospice': 1846,'Paco Sanchez Park': 1847,
                  'Porter Wound Care Center': 1848,'Compu Clases': 1849, 'Kinnear Park': 1850,'North End Park': 1851,'Orient Heights Beach': 1852,'GSVlabs': 1853,'Condos at 326 N Ellsworth Ave': 1854,
                  'Race Recycling and Compacting Equipment': 1855,'Beaver Pond Natural Area On Tc': 1856, 'New Century Hospice': 1857,'Historic Rose Garden': 1858, 'Laura\'s Alteration & Tailor Express': 1859,
                  'Cheesman Park': 1860,'Msgr Reynolds Playground': 1861,'Speer Blvd Park': 1862,'Lombardi Memorial Park': 1863,'Baywood Elementary': 1864,'Bonnie Brae Park': 1865,'Badging Office': 1866,
                  'Barrett Park': 1867,'Jefferson Square Park': 1868,'Harlem Children\'s Zone': 1869,'Niver Creek Trail Greenway': 1870,'Marky\'s on Madison': 1871, 'Goldrick': 1872, 'Fuller Park': 1873,
                  'Savin Hill Beach': 1874,'Majestic Heightz': 1875,'Bloxton hotels': 1876,'Army National Guard': 1877,'Bay Area Disc Centers': 1878,'Clock Repair Team': 1879,'Apartments at 815 E 4th Ave': 1880,
                  'Henry Brooks Senior Housing': 1881,'Old North Memorial Garden': 1882,'Wabash Trailhead': 1883,'First Hill Park': 1884,'VCA Animal Hospitals': 1885,'University Lake Shore Place': 1886,
                  'Fan Pier Park': 1887,'Mattahunt School Woods': 1888,'Boston Architectural College': 1889,'Kelly Open Space': 1890,'El Ciro Fish Market': 1891,'Froula Playground': 1892,'Zuni Park': 1893,
                  'Sandhofer Reservoir': 1894,'Aspgren Clifford Park': 1895,'Cambridge College Boston': 1896,'Boston Children\'s Museum': 1897,'Denver Outdoor Adventure Company': 1898,
                  'RMD Goodyear Tire Center': 1899,'Haven Avenue': 1900,'Fred Thomas Park': 1901,'Ash Grove Park': 1902,'Freedom Debt Relief': 1903,'Simon Poon At Maison Salon': 1904,'LGBT Memorial': 1905,
                  'Washington Trust For Historic Preservation': 1906,'Carleton Court Dog Park': 1907,'Mattahunt Schoolyard': 1908,'Little Boxcar Dog Park': 1909,'Rocky Mountain Lake Park': 1910,
                  'Cherry Tree Park and Playground': 1911,'Homer Harris Park': 1912,'Mamak Cart': 1913,'Manhattan By Sail': 1914,'American Legion Playground': 1915,'Rotella Park': 1916,'Mcconnell Park': 1917,
                  'Hungarian Freedom Park': 1918,'Moscow On The Hudson': 1919,'Sterling Square': 1920,'Channel Center Dog Park': 1921,'Boston China Town Neighborhood Center': 1922,'Gansevoort Plaza': 1923,
                  'Agassiz Road': 1924,'Park Carlton': 1925,'Firehouse Mini Park': 1926,'Drug Detox Centers Denver': 1927,'Hershey\'s Chocolate World': 1928,
                  'Adventist University Of Health Sciences Denver Campus': 1929,'Kunsmiller': 1930,'Hampden Heights North': 1931,'Dent Artist Paintless Dent Removal': 1932,
                  'Proline Auto Body Service': 1933,'Kings Chapel Burying Ground': 1934,'Georges Island': 1935,'Seattle Metropolitan Police Museum': 1936,'Symphony Community Park': 1937,
                  'Museum Of American Gangster': 1938,'Denver Rehabilitation Clinic': 1939,'Swansea': 1940,'Articulated Wall by Herbert Bayer': 1941,'Pinehurst Park': 1942,'Kittredge Park': 1943,
                  'Wings Over The Rockies Air & Space Museum': 1944,'Edith Macefield\'s House': 1945,'Bow Ding Seafood': 1946,'King School Park': 1947,'Can\'t Look Away The Lure of Horror Film Exhibit': 1948,
                  'Byers Evans House Museum': 1949,'Dribbl Basketball': 1950,'HunterS Point Park': 1951,'Keycorp': 1952,'Mosaic Market': 1953,'Spaces': 1954,'Columbia University Inst Real Estate': 1955,
                  'La Alma Lincoln Park': 1956,'Dr Gene Mahaffey': 1957,'Quick Lenore B Park': 1958,'Berkeley Lake Park': 1959,'Gates Planetarium': 1960,'Walker Madame Cj Park': 1961,
                  'Seattle Children\'s Playgarden': 1962,'Me Kwa Mooks Natural Area': 1963,'Thermador Repair Group Industrial Disctrict': 1964,'Your Choice Appliance Service': 1965,
                  'Chinese Community Bulletin Board': 1966, 'Lilac Adult Family Home': 1967,'American Friends of Nishmat': 1968,'Delimanjoo': 1969,'Museum Of Mathematics': 1970,
                  'Colorado State University': 1971, 'S & A 96 Candy Store': 1972, 'Cesar Chavez Park': 1973, 'Jeep Jones Park': 1974, 'Poplar Street Play Area': 1975, 'Rutland Washington Community Garden': 1976,
                  'Creative Recreational Systems Bridgewater': 1977, 'Downtown Seattle Masters': 1978, 'Dewey Square Plaza': 1979,'Colorado Military History Museum': 1980, 'Combs Junior College': 1981,
                  'University House Resident Council': 1982,'Fremont Appliance Refit': 1983,'Harlem Shambles': 1984, 'Westlake Greenbelt': 1985,'Foster Island Wash Park': 1986, 'Zion Learning Center': 1987,
                  'Federal Reserve Plaza': 1988,'Haller Lake Community St End': 1989,'Savin Maywood Street Garden': 1990, 'Westlake Park': 1991, 'City Of Brest Park': 1992,
                  'Highlands Integrative Pediatrics': 1993,'Pike Place Urban Garden': 1994,'Mckinley Thatcher': 1995, 'Golden Triangle': 1996,'St Nicholas Park': 1997, 'Roanoke Street Mini Park': 1998,
                  'Laruleta Meat Mkt': 1999,'Rainier Valley Historical Society': 2000,'Sacred Garden Healing Center': 2001, 'Fred Lind Manor': 2002,'Broadway Hill Park': 2003, 'Capitol Hill Reservoir': 2004,
                  'Southmoor Park': 2005,'Museum Views': 2006,'Queen Anne Upholstery and Refinishing': 2007, 'Clyfford Still Museum': 2008,'9th Street Historic Park': 2009, 'Carla Madison Rec Center': 2010,
                  'Henry Art Gallery': 2011, 'Veracom Ford': 2012, 'NuevaCare': 2013, 'Community Psychiatry': 2014, 'First Republic Bank': 2015, 'Savin Hill Cove': 2016, 'Daggett Museum For Living Artists': 2017,
                  'Borough Of Manhattan Community College Of The City University Of New York': 2018, 'Odyssey': 2019, 'Elite Private Yachts New York': 2020, 'Blue Dog Pond': 2021,
                  'Montclair Recreation Center': 2022,'Empower Field at Mile High': 2023, 'Buena Vista': 2024,'Collyer Brothers Park': 2025,'Aesop Park': 2026,'Dorchester Heights Monument': 2027,
                  'The Denver Hospice': 2028, 'Jenny Manes Premier Bridal Design & Alterations': 2029, 'Campana Quetzal': 2030, 'Lewis Home Care Agency': 2031,'Meat On A Stick': 2032,'PAR3 Communications': 2033,
                  'Fenway Victory Gardens': 2034, 'Bright Landry Hockey Center': 2035,'A Night at the Museum': 2036, 'Southwest Recreation Center': 2037,'Valdez': 2038,'Centura Rehabilitation': 2039,
                  'Hubbard Homestead Park': 2040, 'Gourmet Garage': 2041,'Westside Market NYC': 2042, 'Massachusetts Avenue Malls': 2043, 'Reliable Corporation Lobby Newstand': 2044, 'Thompson Island': 2045,
                  'Lincoln Center Plaza': 2046, 'Wilis Case Golf Course': 2047,'Chef Recipes': 2048, 'Vanderbilt Park': 2049, 'Habitat Park': 2050, 'Broadway Shoe Repair': 2051, 'Elite Vend': 2052,
                  'Cherry Hill Fountain': 2053, 'Malatesta Woodworks': 2054,'Boston University School Of Medicine': 2055,'Lamps Plus': 2056,'Beacon Alterations': 2057, 'Rainbow Point': 2058,
                  'Greenwood Phinney Ruv': 2059, 'Wings Over Washington': 2060,'Alvin Larkins Park': 2061,'A & J Commissary': 2062,'Square Square': 2063, 'Belleview Suites at DTC': 2064,
                  'Huntington Learning Centers': 2065, 'Franklin Hill Bha Court': 2066,'City University Of Seattle': 2067, 'Feel Good Llc': 2068, 'Godiva Cafe': 2069, 'Boston University Hillel': 2070,
                  'Edna V Bynoe Park': 2071, 'Falcon Park': 2072, 'Paul Agustinovich PhD': 2073,'Left Coast Entertainment': 2074,'86 East Thirty Ninth': 2075,'Magnolia Care': 2076, 'Northgate Park': 2077,
                  'Emeritus At Pinehurst Park': 2078, 'Chocolat Moderne': 2079, 'Boston University Plastic & Reconstructive Surgery': 2080, 'Bearing Institute': 2081, 'Havey Beach': 2082,
                  'Horatio Harris Park': 2083, 'Colorado Access': 2084, 'Hiram M Chittenden Locks': 2085,'Confluence East Park': 2086, 'Abingdon Square': 2087, 'New Freedom Park': 2088,
                  'South Street Community Garden': 2089,'Pacific Maritime Institute': 2090, 'Hobby Lobby Stores': 2091,'John C Little Sr Park': 2092, 'Spuyten Duyvil Shorefront Park': 2093,
                  'Brophy Park': 2094, 'Meghna Grocery & Halal Meat': 2095, 'Green Lake Small Craft Center': 2096, 'Pleasant Adult Family Home': 2097,'Public Garden': 2098, 'Wheel Fun Rentals Berkeley Park': 2099,
                  'Larry Bird Plaque': 2100, 'Seattle Asian Art Museum': 2101, 'ChildrenS Park': 2102, 'T Mobile Park': 2103,'Force': 2104, 'Museum of African American History': 2105, 'NB Beauty': 2106,
                  'Williams Dr Daniel Hale Park': 2107, 'Cooper Park Dog Run': 2108, 'Integrity Motors': 2109, 'Itamae Japanese Sushi': 2110, 'College Street Park': 2111, 'Cunningham Park': 2112,
                  'Sunnyside Ave N Boat Ramp': 2113, 'Waterfalls in Central Park': 2114, 'HomeWell Senior Care': 2115,'Ballard Corners Park': 2116, 'Papabubble': 2117, 'Cherry Creek Park': 2118 }

  tl_reverse_mapping = {v: k for k, v in tl_mapping.items()}
  tl_labels = list(tl_mapping.keys())

  # Collecting user inputs
  def get_DAYOFWEEK():
    DAY_OF_WEEK = st.selectbox('Select a day of week 📆', wd_mapping)
    return DAY_OF_WEEK

  def get_TRUCK_BRAND_NAME():
      TRUCK_BRAND_NAME = st.selectbox('Select a truck brand name 🚐', bn_mapping)
      return TRUCK_BRAND_NAME
    
  def get_CITY():
      CITY = st.selectbox('Select a city 🏕️', ct_mapping)
      return CITY

  def get_LOCATION():
      LOCATION = st.selectbox('Select a truck location 📍', tl_mapping)
      return LOCATION  

  # Define the user input fields
  wd_input = get_DAYOFWEEK()
  bn_input = get_TRUCK_BRAND_NAME()
  ct_input = get_CITY()
  tl_input = get_LOCATION()
  
  # Map user inputs to integer encoding
  wd_int = wd_mapping[wd_input]
  bn_int = bn_mapping[bn_input]
  ct_int = ct_mapping[ct_input]
  tl_int = tl_mapping[tl_input]

  if st.button('Predict Daily Sales'):
    # Make the prediction  
    input_data = [[wd_int, bn_int, ct_int, tl_int]]
    input_df = pd.DataFrame(input_data, columns=['DAY_OF_WEEK', 'TRUCK_BRAND_NAME', 'CITY', 'LOCATION'])
    prediction = xgb_alethea.predict(input_df)   

    output_data = [wd_int, bn_int, ct_int, tl_int, prediction[0]]
    output_df = pd.DataFrame([output_data], columns=['DAY_OF_WEEK', 'TRUCK_BRAND_NAME', 'CITY', 'LOCATION', 'DAILY_SALES'])
    predicted_sales = output_df['DAILY_SALES'].iloc[0]
    st.write('The predicted daily sales is {:.2f}.'.format(predicted_sales))

  
  st.divider()
  st.title('Projected Yearly Revenue 💰')
  st.write('Sales in a city are significantly influenced by the level of urban activity, with population size being a key factor directly correlated to daily sales. :green[As city population increases, it tends to drive higher daily sales due to increased consumer demand.] Therefore, leveraging the city\'s population data, we can predict future sales trends, considering the average yearly population growth for each city in the United States of America, as reported in online sources. :blue[With this, we would be able to look at the projected sale increase for each truck brand in teh cities over a span of 5 years]')
  st.write('Average yearly population growth for each city 🌇')
  st.write('San Mateo: 4,600')
  st.write('Seattle: 30,000')
  st.write('New York City: 70,000')
  st.write('Boston: 17,000')
  st.write('Denver: 34,000')

  # Load the cleaned and transformed dataset
  df2 = pd.read_csv('fs_alethea.csv')
  df3 = pd.read_csv('yr_alethea.csv')
  
  # Viewing predicted yearly sales in the future
  def get_Extra():
    YEARS = st.slider('Number of years later 📆', 1, 5, 1)
    st.write("Predicting yearly revenue in ", YEARS, 'year(s)')
    return YEARS  
  et_input = get_Extra()

  originalCity = ct_reverse_mapping[ct_int]
  originalBrand = bn_reverse_mapping[bn_int]
  
  # st.write('Current yearly revenue: {:.2f}.'.format(predicted_sales))
  if st.button('Predict Yearly Revenue'):
    # filtered_data = df2[(df2['CITY'] == ct_int) & (df2['TRUCK_BRAND_NAME'] == bn_int)]
    revenueRow = df3[(df3['CITY'] == originalCity) & (df3['TRUCK_BRAND_NAME'] == originalBrand)]
    # Display predicted revenue
    # predicted_revenue = filtered_data[str(et_input)].iloc[0]

    # Check if the revenueRow DataFrame is not empty
    if not revenueRow.empty:
      # Extract information from the filtered row
      years_revenue = revenueRow['YEARS_REVENUE'].values[0]
      city_pop = revenueRow['CITY_POPULATION'].values[0]
      city = revenueRow['CITY'].values[0]
      truck_brand = revenueRow['TRUCK_BRAND_NAME'].values[0]
      st.write(f'In 2022, the yearly revenue of {truck_brand} in {city} to date is ${years_revenue}, with a population of {city_pop}.')
  
    st.write(f"The projected yearly revenue then would be: :green[predicted rev], with an increase of :green[%] since 2022.")
    
  

with tab2:
  st.title('My Parents New Healthy Diner')
