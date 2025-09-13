from transformers import pipeline  # type: ignore[reportPrivateImportUsage]

print("Question detection examples:")

question_classifier = pipeline(
    "text-classification", model="huaen/question_detection", device="cuda"
)

response = question_classifier(
    """Have you ever pondered the mysteries that lie beneath the surface of every day life?"""
)
print(response)

response = question_classifier(
    """"Life is a journey that must be traveled, no matter how bad the roads and accomodations." - Olive GoldSmith"""
)
print(response)

print("Language detection example:")

language_classifier = pipeline(
    "text-classification",
    model="papluca/xlm-roberta-base-language-detection",
    device="cuda",
)

response = language_classifier("日本の桜は美しいです。")
print(response)

print("Spam classification examples:")

spam_classifier = pipeline(
    "text-classification", model="Delphia/twitter-spam-classifier", device="cuda"
)

response = spam_classifier(
    """Congratulations! You've been selected as the winener of our exclusive prize draw.
    Claim your reward now by clicking on the link below!"""
)
print(response)

response = spam_classifier(
    """Hi Jimmy, I hope you're doing well. I just wanted to remind
    you about our meeting tomorrow at 10 AM in conference room A.
    Please let me know if you have any questions or need any
    further information. Looking forward to seeing you there!"""
)
print(response)

print("Text generation example:")

generator = pipeline("text-generation", model="openai-community/gpt2", device="cuda")
print(generator("In this course, we will teach you how to"))
print(
    generator(
        "In this course, we will teach you how to",
        max_length=50,
        num_return_sequences=3,
    )
)

print("Text summarization:")

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """ 
    A quantum computer is a computer that exploits quantum mechanical 
phenomena. At small scales, physical matter exhibits properties of 
both particles and waves, and quantum computing leverages this 
behavior using specialized hardware. Classical physics cannot 
explain the operation of these quantum devices, and a scalable 
quantum computer could perform some calculations exponentially 
faster than any modern "classical" computer. In particular, a 
large-scale quantum computer could break widely used encryption 
schemes and aid physicists in performing physical simulations; 
however, the current state of the art is still largely 
experimental and impractical.
     
The basic unit of information in quantum computing is the qubit, 
similar to the bit in traditional digital electronics. Unlike a 
classical bit, a qubit can exist in a superposition of its two 
"basis" states, which loosely means that it is in both states 
simultaneously. When measuring a qubit, the result is a 
probabilistic output of a classical bit. If a quantum computer 
manipulates the qubit in a particular way, wave interference 
effects can amplify the desired measurement results. The design 
of quantum algorithms involves creating procedures that allow a 
quantum computer to perform calculations efficiently.
     
Physically engineering high-quality qubits has proven challenging. 
If a physical qubit is not sufficiently isolated from its 
environment, it suffers from quantum decoherence, introducing noise 
into calculations. National governments have invested heavily in 
experimental research that aims to develop scalable qubits with 
longer coherence times and lower error rates. Two of the most 
promising technologies are superconductors (which isolate an 
electrical current by eliminating electrical resistance) and ion 
traps (which confine a single atomic particle using electromagnetic 
fields).
    
Any computational problem that can be solved by a classical computer 
can also be solved by a quantum computer.[2] Conversely, any problem 
that can be solved by a quantum computer can also be solved by a 
classical computer, at least in principle given enough time. In other 
words, quantum computers obey the Church–Turing thesis. This means 
that while quantum computers provide no additional advantages over 
classical computers in terms of computability, quantum algorithms 
for certain problems have significantly lower time complexities than 
corresponding known classical algorithms. Notably, quantum computers 
are believed to be able to solve certain problems quickly that no 
classical computer could solve in any feasible amount of time—a feat 
known as "quantum supremacy." The study of the computational 
complexity of problems with respect to quantum computers is known as 
quantum complexity theory.
    """

print("Without sampling...:")
print(
    summarizer(article, min_length=100, max_length=250, do_sample=False)
)  # extractive summarization
print("...and with sampling:")
print(summarizer(article, min_length=100, max_length=250, do_sample=True))

print("Text translation:")

translator = pipeline(
    task="translation_en_to_fr", model="google-t5/t5-base", device="cuda"
)
print(
    translator(
        "Wikipedia is hosted by the Wikimedia Foundation, a non-profit organization that also hosts a range of other projects."
    )
)
translator = pipeline(
    task="translation_en_to_de", model="google-t5/t5-base", device="cuda"
)
print(
    translator(
        "Wikipedia is hosted by the Wikimedia Foundation, a non-profit organization that also hosts a range of other projects."
    )
)

translator = pipeline(
    "translation_en_to_zh", model="facebook/m2m100_418M", device="cuda"
)
print(
    translator(
        "Wikipedia is hosted by the Wikimedia Foundation, a non-profit organization that also hosts a range of other projects."
    )
)
translator = pipeline(
    task="translation_zh_to_en", model="facebook/m2m100_418M", device="cuda"
)
print(
    translator("维基百科是维基百科基金会主办的,是一家非营利组织,还主办了许多其他项目。")
)
