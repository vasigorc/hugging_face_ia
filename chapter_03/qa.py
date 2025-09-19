import pandas as pd
from transformers import pipeline  # type: ignore[reportPrivateImportUsage]

QA_model = pipeline(
    task="question-answering", model="deepset/roberta-base-squad2", device="cuda"
)

# a text paragraph discussing the origin of the Singapore name
text = """
    The English name of "Singapore" is an anglicisation of the native 
Malay name for the country, Singapura (pronounced [siŋapura]), 
    which was in turn derived from the Sanskrit word for 'lion city' 
(Sanskrit: सिंहपुर; romanised: Siṃhapura; Brahmi: 𑀲𑀺𑀁𑀳𑀧𑀼𑀭; literally 
    "lion city"; siṃha means 'lion', pura means 'city' or 'fortress' ).
    Pulau Ujong was one of the earliest references to Singapore Island, 
    which corresponds to a Chinese account from the third century 
                                            referred to a place as Pú Luó Zhōng (Chinese: 蒲 羅 中), a 
                                                                                    transcription of the Malay name for 'island at the end of a 
peninsula'. Early references to the name Temasek (or Tumasik) are 
found in the Nagarakretagama, a Javanese eulogy written in 1365, 
and a Vietnamese source from the same time period. The name possibly 
    means Sea Town, being derived from the Malay tasek, meaning 'sea' or 
'lake'. The Chinese traveller Wang Dayuan visited a place around 1330 
        named Danmaxi (Chinese: 淡馬錫; pinyin: Dànmǎxí; Wade–Giles: Tan Ma Hsi) 
    or Tam ma siak, depending on pronunciation; this may be a transcription 
    of Temasek, alternatively, it may be a combination of the Malay Tanah
    meaning 'land' and Chinese xi meaning 'tin', which was traded on the 
island
"""

question = "What is the meaning of Singapura?"
question_with_context = {"question": question, "context": text}

model_response = QA_model(question_with_context)
print("=" * 50)
print(f"Original text:\n {text}")
print("-" * 50)
print(f"Question: {question}")
print("-" * 50)
print("Model response:")
print(pd.DataFrame([model_response]))
print("=" * 50)
