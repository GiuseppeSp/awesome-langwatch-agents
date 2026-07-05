"""
The tool registry — 101 tools with many NEAR-DUPLICATE descriptions.

20 of the tools are the correct answer to some query in the dataset; the rest are
distractors, many of them close neighbors of the correct ones (exchange_rate near
convert_currency, uv_index/pollen_count near weather, market_index near stock_price,
map_directions near distance_between, ...). At this scale, selecting from ALL tools
means the model reads 100+ descriptions and must disambiguate among close neighbors
— exactly the condition under which pre-filtering (retrieval) is supposed to help.

The registry started at 30 tools (both modes scored 100% — no headroom); it was
padded to ~100 to test whether a genuinely large tool set degrades all-tools
selection. Each tool has a name, a one-line description, and (for the answer tools)
a deterministic mock output.
"""

from __future__ import annotations

TOOLS: dict[str, str] = {
    "convert_currency": "Convert an amount of money from one currency to another using current rates.",
    "convert_units": "Convert a physical quantity between measurement units (length, weight, temperature).",
    "convert_timezone": "Convert a time from one timezone to another.",
    "translate_text": "Translate a piece of text from one language to another.",
    "transliterate_text": "Convert text from one script/alphabet to another without translating meaning.",
    "summarize_text": "Produce a short summary of a longer passage of text.",
    "sentiment_analysis": "Classify the emotional sentiment (positive/negative/neutral) of a text.",
    "detect_language": "Identify which natural language a piece of text is written in.",
    "spellcheck": "Find and correct spelling mistakes in a text.",
    "word_count": "Count the number of words in a text.",
    "stock_price": "Look up the current share price of a publicly traded company.",
    "crypto_price": "Look up the current price of a cryptocurrency.",
    "company_info": "Look up general profile information about a company (industry, HQ, size).",
    "weather_current": "Get the current weather conditions for a location.",
    "weather_forecast": "Get the multi-day weather forecast for a location.",
    "air_quality": "Get the current air quality index for a location.",
    "geocode_address": "Convert a street address into latitude/longitude coordinates.",
    "reverse_geocode": "Convert latitude/longitude coordinates into a street address.",
    "distance_between": "Compute the travel distance between two locations.",
    "flight_status": "Look up the current status of a specific flight by number.",
    "flight_search": "Search for available flights between two airports on a date.",
    "hotel_search": "Search for available hotels in a city for given dates.",
    "restaurant_search": "Search for restaurants near a location.",
    "calendar_create_event": "Create a new event on the user's calendar.",
    "calendar_list_events": "List upcoming events from the user's calendar.",
    "send_email": "Compose and send an email to a recipient.",
    "send_sms": "Send a text message to a phone number.",
    "set_reminder": "Set a reminder for a future time.",
    "unit_price": "Compute the price per unit given a total price and quantity.",
    "tip_calculator": "Compute a tip amount and total for a restaurant bill.",
    # --- distractor tools: pad the registry to ~100 to simulate a LARGE tool set ---
    "convert_temperature": "Convert a temperature between Celsius, Fahrenheit and Kelvin.",
    "convert_data_size": "Convert a data size between bytes, KB, MB, GB.",
    "convert_speed": "Convert a speed between km/h, mph and m/s.",
    "percentage_calc": "Compute a percentage of a number.",
    "loan_calculator": "Compute monthly payments on a loan.",
    "bmi_calculator": "Compute body mass index from height and weight.",
    "age_calculator": "Compute a person's age from their birth date.",
    "date_difference": "Compute the number of days between two dates.",
    "paraphrase_text": "Rewrite a text to say the same thing differently.",
    "grammar_check": "Find and correct grammar mistakes in a text.",
    "keyword_extract": "Extract the main keywords from a text.",
    "text_to_speech": "Convert written text into spoken audio.",
    "speech_to_text": "Transcribe spoken audio into written text.",
    "profanity_filter": "Detect and mask profane words in a text.",
    "readability_score": "Compute the reading difficulty of a text.",
    "named_entity_recognition": "Extract names of people, places and organizations from text.",
    "exchange_rate": "Look up the exchange rate between two currencies.",
    "dividend_history": "Look up the dividend payment history of a stock.",
    "market_index": "Look up the current value of a market index like the S&P 500.",
    "mortgage_rate": "Look up current mortgage interest rates.",
    "invoice_generate": "Generate an invoice from line items.",
    "expense_report": "Compile an expense report from receipts.",
    "uv_index": "Get the current UV index for a location.",
    "pollen_count": "Get the current pollen levels for a location.",
    "sunrise_sunset": "Get sunrise and sunset times for a location.",
    "tide_times": "Get the tide schedule for a coastal location.",
    "elevation_lookup": "Get the elevation above sea level for coordinates.",
    "map_directions": "Get turn-by-turn directions between two places.",
    "nearby_places": "Find points of interest near a location.",
    "timezone_lookup": "Get the timezone for a given location.",
    "car_rental_search": "Search for rental cars in a city.",
    "train_schedule": "Look up train departure times between stations.",
    "public_transit": "Get public transit routes between two stops.",
    "visa_requirements": "Look up visa requirements for a destination.",
    "currency_of_country": "Look up which currency a country uses.",
    "baggage_rules": "Look up an airline's baggage allowance.",
    "seat_map": "Show the seat map for a flight.",
    "send_slack_message": "Post a message to a Slack channel.",
    "create_task": "Create a to-do task in a task manager.",
    "calendar_delete_event": "Delete an event from the user's calendar.",
    "calendar_find_slot": "Find a free time slot in the user's calendar.",
    "contact_lookup": "Look up a contact's details by name.",
    "note_create": "Create a note in the user's notes app.",
    "poll_create": "Create a poll to collect votes.",
    "meeting_transcribe": "Transcribe a meeting recording.",
    "qr_generate": "Generate a QR code encoding a URL or text.",
    "barcode_lookup": "Look up a product by its barcode.",
    "image_resize": "Resize an image to given dimensions.",
    "image_ocr": "Extract text from an image.",
    "pdf_merge": "Merge several PDF files into one.",
    "pdf_split": "Split a PDF into separate pages.",
    "password_generate": "Generate a strong random password.",
    "hash_text": "Compute a cryptographic hash of a text.",
    "uuid_generate": "Generate a unique identifier.",
    "color_convert": "Convert a color between hex, RGB and HSL.",
    "random_number": "Generate a random number in a range.",
    "dice_roll": "Roll dice and return the result.",
    "recipe_search": "Search for cooking recipes by ingredient.",
    "nutrition_lookup": "Look up the nutrition facts of a food.",
    "workout_plan": "Generate a workout plan.",
    "movie_info": "Look up information about a movie.",
    "book_info": "Look up information about a book.",
    "song_identify": "Identify a song from a short audio clip.",
    "define_word": "Look up the dictionary definition of a word.",
    "synonym_lookup": "Find synonyms for a word.",
    "wikipedia_summary": "Get a summary of a Wikipedia topic.",
    "news_headlines": "Get the latest news headlines for a topic.",
    "sports_score": "Look up the score of a sports match.",
    "horoscope": "Get today's horoscope for a star sign.",
    "joke_generator": "Return a random joke.",
    "fact_generator": "Return a random interesting fact.",
}

# Deterministic mock outputs, keyed by tool. The value is what "calling" the tool returns.
_OUTPUTS: dict[str, str] = {
    "convert_currency": "230.00 EUR",
    "convert_units": "6.21 miles",
    "convert_timezone": "8:00 PM London time",
    "translate_text": "おはよう (ohayou)",
    "detect_language": "French",
    "summarize_text": "(a 2-sentence summary)",
    "sentiment_analysis": "positive",
    "stock_price": "142.50 USD",
    "crypto_price": "61,240 USD",
    "weather_current": "12C, cloudy",
    "weather_forecast": "rain expected on days 2 and 4",
    "air_quality": "AQI 168 (unhealthy)",
    "geocode_address": "51.5034, -0.1276",
    "flight_status": "BA249 on time, departs 18:40",
    "hotel_search": "3 hotels found",
    "calendar_create_event": "event created for Friday",
    "send_sms": "message sent",
    "tip_calculator": "tip 16.80, total 100.80",
    "distance_between": "225 km",
    "set_reminder": "reminder set for 9:00 AM tomorrow",
}


def call_tool(name: str) -> str:
    """Execute the (mock) tool and return its result, or an error for unknown tools."""
    if name not in TOOLS:
        return f"ERROR: no tool named '{name}'"
    return _OUTPUTS.get(name, "(ok)")
