import os
import time
import random
from datetime import datetime
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

# --- CONFIGURATION ---
USERNAME = os.environ.get("IG_USERNAME", "cozycorner4245")
PASSWORD = os.environ.get("IG_PASSWORD")
IMAGE_PATH = os.path.abspath("generated_post.jpg")
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

def generate_ai_cozy_image(output_path):
    """Generates a theme-specific image using 1 of 50 randomized cozy prompts."""
    print("Initializing AI Image Generator...")
    
    # 50 Highly detailed, curated Cozy Corner thematic prompts
    cozy_prompts = [
        # --- Warm Living Spaces (1-10) ---
        "A quiet cozy room corner, warm soft lighting, a beige armchair with a fluffy blanket, a hot steaming mug on a wooden side table, cinematic aesthetic, warm tones, high quality",
        "Sunlight filtering through sheer white curtains onto a light oak floor, a woven floor cushion, a small ceramic cup of coffee emitting soft steam, minimalist cozy aesthetic",
        "An indoor macrame hammock draped with a soft knit throw blanket, warm fairy lights twinkling in the background, bohemian comfort vibes, pastel grading",
        "A small rustic reading nook, a velvet floor pillow, a burning candle with a soft golden flame on a tray, a soft focused stack of old books, warm ambient glow",
        "A plush linen sofa strewn with mismatched neutral-toned pillows, a low wooden coffee table with a small clear vase holding dried pampas grass, calming minimalist interior",
        "A vintage leather armchair pushed into a sunlit alcove, a delicate green fern hanging beside it, soft sunbeams highlighting dust motes in the warm air",
        "A close-up of a rustic wooden bench next to an indoor fireplace, a cozy thick plaid blanket folded neatly on top, flickering amber lighting",
        "A boho style corner with cream-colored rugs, a textured floor pouf, a small ceramic bowl filled with dried lavender, soft afternoon shadows",
        "A warm loft balcony corner filled with soft throw rugs, a small wooden stool supporting a single ceramic matcha bowl, soft cinematic morning mist",
        "An oversized knitted beanbag chair in a bright cream-colored studio room, gentle ambient lighting, a single green monstera leaf casting a soft shadow",

        # --- Window Views & Weather Vibes (11-20) ---
        "A peaceful window corner on a rainy day, rain droplets visible on glass, fairy lights softly glowing, a stack of books next to a cup of chamomile tea, soft focus",
        "A frosty windowpane looking out into a quiet snow-covered pine forest, inside a steaming glass mug sits on a wooden ledge next to a knit scarf, cozy winter contrast",
        "An open window sill during a breezy spring morning, soft pink cherry blossom petals on the ledge, a delicate white tea set, pastel aesthetic",
        "An autumn window view with bright golden leaves falling outside, on the inside ledge sits a miniature pumpkin and a steaming cup of cider, warm sepia tones",
        "A cozy window seat at dusk, the sky outside a deep twilight purple, inside a small lantern glows softly next to a tartan wool blanket",
        "A wide bay window filled with plush cream cushions, a heavy rainstorm outside blurring the city lights, indoor string lights casting a warm contrast",
        "Morning golden hour hitting a rustic wooden window frame, a ceramic pot with a small sprout sitting in the warm light, peaceful lofi vibes",
        "A close-up of a window ledge at sunrise, soft orange hues filling the sky outside, a clear glass mug of hot water with lemon casting shadows",
        "A window seat draped in fuzzy sheepskin rugs, looking out onto a misty green mountain valley, calm isolated comfort theme",
        "A modern window frame overlooking a quiet starry night sky, a small candle glowing softly on the dark wood sill inside, deep calm ambiance",

        # --- Desks, Studies & Creative Corners (21-30) ---
        "An aesthetic minimalist desk setup, warm desk lamp glow, a small indoor plant, an open blank linen notebook, a scented candle burning softly, clean composition",
        "A close up of hands wrapped around a rustic ceramic mug filled with dark coffee, sitting next to an open sketchbook on a grainy oak table, artisan aesthetic",
        "An old wooden study desk cluttered with vintage fountain pens, small dried flower bookmarks, and an amber glass jar, soft cinematic shadows",
        "A low-profile lofi study setup, a glowing laptop screen open to a peaceful scenery, a small desk plant, warm string lights draped overhead",
        "A pottery studio corner, beautifully unpolished ceramic clay mugs drying on a wooden shelf, soft sunlight coming from a high window, artisan comfort",
        "A beautifully organized creative desk, watercolors laid out on a tray, an aesthetic ceramic brush holder, soft cream and beige color palette",
        "An open antique book on a soft duvet, a pair of brass wire-rimmed glasses resting on the page, soft morning light hitting the paper",
        "A cozy writer's corner, an old-fashioned typewriter on a dark wooden table, a single small porcelain teacup emitting faint steam",
        "A bedside table setup, a stack of three hardback books, a small ceramic dish holding rings, a warm spherical lamp softly lit",
        "A minimalist crafting table, skeins of soft chunky yarn in pastel pink and beige sitting in a wicker basket, soft diffused lighting",

        # --- Coffee, Tea & Cafe Comforts (31-40) ---
        "A close-up of dynamic latte art in a wide ceramic mug, sitting on a textured linen napkin on a sunlit cafe table, beautiful aesthetic composition",
        "A clear glass teapot filled with blooming herbal tea, a small single flame tealight heater underneath it glowing softly, minimalist glass aesthetic",
        "A wooden tray resting on a soft white bed, holding a porcelain plate with a flaky croissant and a small glass of fresh milk, morning luxury vibe",
        "A close-up of a hand pouring warm honey from a wooden dipper into a ceramic tea bowl, rich warm textures, soft lighting",
        "An aesthetic coffee brewing corner, a glass pour-over coffee carafe half full, clean white tiled background, warm wooden accents",
        "A small cast-iron teapot sitting next to a bamboo whisk and a small bowl of green matcha powder, serene zen minimalist aesthetic",
        "A cozy kitchen counter corner, morning sunlight filtering through lace curtains, a single freshly baked cinnamon roll on a ceramic plate",
        "A dark wood coffee table scene, a hot cocoa mug piled high with tiny melting marshmallows, warm glow from a nearby fireplace",
        "A rustic tray in a garden alcove holding a transparent cup of iced tea with mint leaves, dappled sunlight filtering through overhead leaves",
        "A close-up of espresso dripping into a small white ceramic cup, steam rising elegantly against a dark moody background",

        # --- Nature, Bedding & Soft Textures (41-50) ---
        "An aesthetic bedroom corner, a bed layered with heavy linen duvets in eucalyptus green and soft white, string lights headboard glow",
        "A close-up of a small green succulent growing out of a tiny handmade clay pot, sitting on a textured stone coaster, bright crisp lighting",
        "A wicker basket overflowing with rolled up waffle-knit towels and soft cotton blankets, sitting next to a tall indoor fiddle-leaf fig plant",
        "A low floor mattress arrangement with crisp white sheets and a chunky knit chunky blanket, soft boho chic aesthetic, morning light",
        "A rustic stone fireplace mantle decorated with small jars of dried baby's breath flowers and a single glowing amber candle",
        "A close-up texture shot of hand-woven beige macrame fabric hanging against a clean white textured plaster wall, soft side lighting",
        "A warm greenhouse corner, small clay plant pots arranged on wooden steps, a watering can, soft humid sunlight filtering through glass",
        "A bedside scene, a single white rose in a small glass bud vase, a linen book laying open next to it face down, soft focus",
        "A cluster of different sized aesthetic candles burning at different heights on a mirror tray, creating a warm dancing golden glow",
        "A simple white porcelain bowl filled with fresh ripe strawberries sitting on an unbleached linen tablecloth, bright airy morning vibe"
    ]
    
    # Pick a random cozy scene prompt from the 50 options
    selected_prompt = random.choice(cozy_prompts)
    print(f"AI Prompt Selected: {selected_prompt}")

    # Load a free, fast AI text-to-image model
    model_id = "OFA-Sys/small-stable-diffusion-v0-1" 
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    
    # Generate the image
    print("AI is painting your cozy scene... (this takes roughly 60 seconds)")
    ai_image = pipe(selected_prompt, num_inference_steps=20, height=512, width=512).images[0]
    
    # Resize to standard Instagram square format
    ai_image = ai_image.resize((1080, 1080), Image.Resampling.LANCZOS)
    
    # Add page handle overlay gracefully onto the image
    draw = ImageDraw.Draw(ai_image)
    try:
        font = ImageFont.truetype(FONT_PATH, 35)
    except IOError:
        font = ImageFont.load_default()
        
    # Draw dark translucent bar at the bottom for text readability
    draw.rectangle([(0, 1020), (1080, 1080)], fill=(0, 0, 0, 80))
    draw.text((40, 1035), f"✨ @{USERNAME} | Your Daily Comfort", fill="#FFFFFF", font=font)

    # Save the final image
    ai_image.save(output_path, quality=95)
    print(f"AI image saved successfully to {output_path}")

def publish_post(image_path_to_upload):
    """Uses Playwright automation to publish the AI generated image."""
    CAPTION = f"✨ Pull up a chair, grab a warm drink, and settle in. Welcome to your cozy corner. \n.\n.\n#cozycorner #cozyvibes #aesthetic #lofi #mentalpeace #warmth #{datetime.now().strftime('%Y')}"

    if not PASSWORD:
        print("Error: IG_PASSWORD environment variable is missing in GitHub Secrets.")
        return

    with sync_playwright() as p:
        device = p.devices['iPhone 12 Pro']
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(**device)
        page = context.new_page()

        print("Navigating to Instagram...")
        page.goto("https://www.instagram.com/accounts/login/")
        page.wait_for_load_state("networkidle")

        print("Logging in...")
        page.fill("input[name='username']", USERNAME)
        page.fill("input[name='password']", PASSWORD)
        page.locator("input[name='password']").press("Enter")
        page.wait_for_timeout(7000)

        # Handle random popups safely
        try: page.click("text=Not Now", timeout=3000)
        except: pass
        try: page.click("button:has-text('Cancel')", timeout=3000)
        except: pass

        print("Uploading AI Post...")
        page.click("svg[aria-label='New Post']")
        page.wait_for_timeout(3000)

        with page.expect_file_chooser() as file_chooser_info:
            page.click("text=Select From Computer") 
        file_chooser = file_chooser_info.value
        file_chooser.set_files(image_path_to_upload)
        page.wait_for_timeout(5000) 

        page.click("div[role='button']:has-text('Next')")
        page.wait_for_timeout(3000)

        page.locator("textarea[aria-label='Write a caption...']").fill(CAPTION)
        page.wait_for_timeout(2000)

        page.click("div[role='button']:has-text('Share')")
        print("Waiting for upload confirmation...")
        page.wait_for_selector("text=Your post was shared", timeout=30000)

        print("Success! A beautiful new cozy view has been uploaded.")
        browser.close()

if __name__ == "__main__":
    if os.path.exists(IMAGE_PATH):
        os.remove(IMAGE_PATH)
        
    generate_ai_cozy_image(IMAGE_PATH)
    if os.path.exists(IMAGE_PATH):
        publish_post(IMAGE_PATH)
