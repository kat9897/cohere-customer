import cohere
from cohere.classify import Example
from constants import API_KEY

co = cohere.Client(API_KEY)


prompt = f"""Passage: Is Wordle getting tougher to solve? Players seem to be convinced that the game has gotten harder in recent weeks ever since The New York Times bought it from developer Josh Wardle in late January. The Times has come forward and shared that this likely isn't the case. That said, the NYT did mess with the back end code a bit, removing some offensive and sexual language, as well as some obscure words There is a viral thread claiming that a confirmation bias was at play. One Twitter user went so far as to claim the game has gone to "the dusty section of the dictionary" to find its latest words.

TLDR: Wordle has not gotten more difficult to solve.
--
Passage: ArtificialIvan, a seven-year-old, London-based payment and expense management software company, has raised $190 million in Series C funding led by ARG Global, with participation from D9 Capital Group and Boulder Capital. Earlier backers also joined the round, including Hilton Group, Roxanne Capital, Paved Roads Ventures, Brook Partners, and Plato Capital.

TLDR: ArtificialIvan has raised $190 million in Series C funding.
--
Passage: The National Weather Service announced Tuesday that a freeze warning is in effect for the Bay Area, with freezing temperatures expected in these areas overnight. Temperatures could fall into the mid-20s to low 30s in some areas. In anticipation of the hard freeze, the weather service warns people to take action now.

TLDR:"""
examples = [Example("The order came 5 days early", "positive"), 
Example("The item exceeded my expectations", "positive"), 
Example("I ordered more for my friends", "positive"), 
Example("I would buy this again", "positive"), 
Example("I would recommend this to others", "positive"), 
Example("The package was damaged", "negative"), 
Example("The order is 5 days late", "negative"), 
Example("The order was incorrect", "negative"), 
Example("I want to return my item", "negative"), 
Example("The item\'s material feels low quality", "negative")]

inputs = ["This item was broken when it arrived", "This item broke after 3 weeks"]

response = co.classify( 
    model='large', 
    inputs=inputs,
    examples=examples)

print('The confidence levels of the labels are: {}'.format(response.classifications))

