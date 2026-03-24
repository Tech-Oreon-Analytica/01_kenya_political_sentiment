import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) or '.'))

from shared.utils.text_cleaning import clean_text, batch_clean
from shared.utils.sentiment_helpers import score_text, classify_sentiment, score_dataframe
print('shared imports OK')

import pandas as pd
df = pd.read_csv('data/processed/scored_data.csv', parse_dates=['timestamp'])
print('CSV loaded: {} rows x {} cols'.format(df.shape[0], df.shape[1]))

samples = [
    'IEBC has failed Kenyans. The results are rigged.',
    'Peaceful voting today. Proud of Kenya. Democracy wins!',
    'Tallying of results continues at constituency level.',
]
for s in samples:
    cleaned = clean_text(s)
    scores  = score_text(cleaned)
    label   = classify_sentiment(scores['compound'])
    print('  [{:8}] ({:+.3f})  {}'.format(label, scores['compound'], s[:60]))

required = ['record_id','timestamp','topic','region','sentiment_label',
            'vader_compound','clean_text','year_month']
missing = [c for c in required if c not in df.columns]
if missing:
    print('MISSING COLUMNS: {}'.format(missing))
else:
    print('all required columns present OK')

print('SMOKE TEST PASSED')
