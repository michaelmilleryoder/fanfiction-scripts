from fanfiction import Scraper
import pickle
import urllib.request
import pdb

scraper = Scraper()
metadata = []

for i in range(1, 12966170):

    if i % int(1e5) == 0:
        save_metadata(metadata, i)
        metadata = []        

    # Test whether story exists for given ID
    base_url = 'http://fanfiction.net'
    url = '{0}/s/{1}'.format(base_url, i)
    page = str(urllib.request.urlopen(url).read())
    error_messages = ['Unable to locate story. Code 1.',
                    'FanFiction.Net Message Type 1',
                    ]    
    if any(e in page for e in error_messages): 
        print(f"No story found for ID {i}")
        continue
    story_metadata = scraper.scrape_story_metadata(i)
    metadata.append(story_metadata)
    print(i)

    # Get metadata
    #try:
    #    story_metadata = scraper.scrape_story_metadata(i)
    #    metadata.append(story_metadata)
    #    print(i)
    ##except AttributeError as e:
    #except:
    #    print(f"No story found for ID {i}")
    #    continue

def save_metadata(metadata, i):
    # Save out metadata
    out_fpath = f'data/fanfiction-net-metadata_{i}.pkl'
    with open(out_fpath, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"Data saved to {out_fpath}")
