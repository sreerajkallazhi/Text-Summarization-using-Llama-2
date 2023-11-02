from src.huggingface_utils import initialize_huggingface
from src.langchain_utils import initialize_langchain

def main():
    # Initialize Hugging Face pipeline
    pipeline = initialize_huggingface()

    # Initialize LangChain
    llm_chain = initialize_langchain(pipeline)

    # Example texts
    text1 = """ A large language model (LLM) is a type of artificial intelligence (AI) algorithm that uses deep learning techniques and massively large data sets to understand, summarize, generate and predict new content. The term generative AI also is closely connected with LLMs, which are, in fact, a type of generative AI that has been specifically architected to help generate text-based content.

Over millennia, humans developed spoken languages to communicate. Language is at the core of all forms of human and technological communications; it provides the words, semantics and grammar needed to convey ideas and concepts. In the AI world, a language model serves a similar purpose, providing a basis to communicate and generate new concepts.

The first AI language models trace their roots to the earliest days of AI. The Eliza language model debuted in 1966 at MIT and is one of the earliest examples of an AI language model. All language models are first trained on a set of data, and then they make use of various techniques to infer relationships and then generate new content based on the trained data. Language models are commonly used in natural language processing (NLP) applications where a user inputs a query in natural language to generate a result.
 """
    result1 = llm_chain.run(text1)
    print(result1)

    text2 = """ Space Exploration Technologies Corp., commonly referred to as SpaceX, is an American spacecraft manufacturer, launch service provider, defense contractor and satellite communications company headquartered in Hawthorne, California. The company was founded in 2002 by Elon Musk with the goal of reducing space transportation costs and to colonize Mars. The company currently operates the Falcon 9 and Falcon Heavy rockets along with the Dragon spacecraft.

The company offers internet service via its Starlink satellites, which became the largest-ever satellite constellation in January 2020 and as of June 2023 comprised more than 4,300 small satellites in orbit.[7] Starlink was also notably used in the war in Ukraine.[8]

Meanwhile, the company is developing Starship, a human-rated, fully-reusable, super heavy-lift launch system for interplanetary and orbital spaceflight. On its failed first flight in April 2023, it became the largest and most powerful rocket ever flown.

SpaceX is the first private company to develop a liquid-propellant rocket that has reached orbit; to launch, orbit, and recover a spacecraft; to send a spacecraft to the International Space Station; and to send astronauts to the International Space Station. It is also the first organization of any type to achieve a vertical propulsive landing of an orbital rocket booster and the first to reuse such a booster. The company's Falcon 9 rockets have landed and reflown more than 200 times.[9]
 """
    result2 = llm_chain.run(text2)
    print(result2)

    text3 = """" Script choice can also be an indicator of whether a word is
borrowed (a concept introduced by Bali et al. (2014) and
later expanded on by Patro et al. (2017)).

As opposed to code-switching, where the switching is in-
tentional and the speaker is aware that the conversation in-
volves multiple languages, a borrowed word loses its orig-
inal identity and is used as a part of the lexicon of the lan-
guage (Patro et al., 2017). However, as the authors say, it is

very hard to ascertain whether a word is borrowed or not.
We hypothesize that if a word is borrowed from English to
Hindi, it will have a higher propensity of being represented

in the Devanagari script (as opposed to Roman) in mixed-
script tweets in the Hindi context, and vice versa.

For instance, consider these three categories of words,
• Words native to Hindi as a baseline (such as ‘Dharma’
- धमर्)
• English words that are likely borrowed (such as ‘Vote’
- वोट and ‘Petrol’ - पेट्रोल)
• English words that are not likely borrowed (such as
‘Minister’ - िमिनस्टर)
We measure the propensity of these words being written in
Devanagari by calculating the ratio of their frequencies in

the two scripts. Ps(w) is the propensity of the word w be-
ing written in script s which, in our case, is equal to the

frequency of w in s in the mixed-script tweet """
    result3 = llm_chain.run(text3)
    print(result3)

if __name__ == "__main__":
    main()
