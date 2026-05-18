import os
import random

import torch
import transformers  # noqa: F401
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright


def _click_first_available(page, selectors, timeout=6000):
    for selector in selectors:
        try:
            page.wait_for_selector(selector, timeout=timeout)
            page.click(selector, timeout=timeout)
            print(f"Clicked selector: {selector}")
            return True
        except Exception:
            continue
    return False


def _upload_image(page, absolute_image_path, timeout=20000):
    upload_selector = "input[type='file']"

    try:
        page.wait_for_selector(upload_selector, state="attached", timeout=timeout)
        page.set_input_files(upload_selector, absolute_image_path, timeout=timeout)
        print("[POST] Uploaded image via direct file input")
        return
    except Exception as direct_upload_error:
        print(f"[POST] Direct file input upload unavailable, falling back to chooser: {direct_upload_error}")

    with page.expect_file_chooser(timeout=timeout) as chooser_info:
        clicked_upload_trigger = _click_first_available(
            page,
            [
                "text=Select from computer",
                "text=Select From Computer",
                "button:has-text('Select from computer')",
                "button:has-text('Select From Computer')",
                "div[role='button']:has-text('Select from computer')",
                "div[role='button']:has-text('Select From Computer')",
            ],
            timeout=10000,
        )
        if not clicked_upload_trigger:
            raise RuntimeError("Upload trigger button was not found.")
    chooser_info.value.set_files(absolute_image_path)
    print("[POST] Uploaded image via file chooser fallback")


def generate_ai_cozy_image(output_path):
    print("[IMAGE] Starting cozy image generation")
    prompts = [
        "A warm reading nook with a plaid blanket, steaming mug of tea, and soft golden lamplight on a rainy evening",
        "Minimalist beige bedroom with neutral tones, linen pillows, morning sunlight filtering through sheer curtains",
        "A rustic wooden desk near a rain-streaked window, open journal, flickering candle, and a cup of coffee",
        "Cozy living room corner with a cream knit throw, houseplants, warm fairy lights, and soft shadows",
        "Scandinavian lounge chair by a frosted window, wool socks, open poetry book, and chamomile tea in a ceramic cup",
        "Small attic corner with slanted roof, amber lantern glow, chunky blanket, and soft rain ambience outside",
        "Muted earth-tone studio apartment nook with floor cushions, incense smoke, and low sunset light",
        "Vintage armchair beside a brick fireplace, stack of classic novels, and a cinnamon latte on a wooden stool",
        "Calm bay window seat with neutral cushions, drizzle outside, string lights overhead, and eucalyptus stems",
        "Hygge-inspired coffee corner with pour-over set, linen runner, warm oak textures, and candlelight",
        "Soft boho reading corner with macrame wall hanging, textured rug, and gentle afternoon sunbeams",
        "A quiet bedroom corner with warm bedside lamp, folded knit cardigan, and a mug of cocoa on a tray",
        "Rainy city evening through tall window, indoor fern, amber candle jar, and an open sketchbook",
        "Cream-toned sofa corner with boucle pillows, small side table, and a teapot releasing visible steam",
        "Cozy minimalist workspace with warm desk lamp, fountain pen, notebook, and rainy bokeh outside",
        "A small sunroom with wicker chair, pastel blanket, tea set, and potted herbs in soft morning light",
        "Rustic cabin interior with wool blanket, crackling fireplace, and hot coffee near a wooden bookshelf",
        "Peaceful winter window nook with snow outside, knit socks, and vanilla candle beside a hardcover book",
        "Warm neutral bedroom with layered linen bedding, dried pampas grass, and soft sunrise glow",
        "Quiet corner bench with plaid throw, ceramic mug, and golden fairy lights reflected on glass",
        "Moody evening study corner with vintage lamp, handwritten notes, and espresso cup on reclaimed wood desk",
        "Soft cottage kitchen corner with fresh pastry, herbal tea, and sunlight on white ceramic tiles",
        "Cozy indoor plant corner with monstera leaves, woven basket, and warm lamp creating gentle shadows",
        "Minimal Japandi corner with low table, matcha bowl, linen cushion, and diffuse daylight",
        "Rainy-day blanket fort aesthetic with books, fairy lights, and marshmallow hot chocolate",
        "Warm-toned living room with wool rug, flickering candles, and a calm lo-fi atmosphere",
        "Small reading alcove under staircase, wooden shelves, amber light, and steaming chai latte",
        "Neutrals-only bedroom corner with boucle chair, soft throw, and morning coffee on a round tray",
        "Cozy balcony corner with lanterns, knit blanket, and cloudy twilight sky in muted pastel tones",
        "Elegant coffee table setup with tea candles, art book, and cappuccino foam beside dried flowers",
        "Quiet rainy afternoon in a window seat, knitted blanket, cat curled up, and soft piano mood",
        "Warm minimal interior with oak stool, ceramic vase, and candle glow against textured plaster wall",
        "Soft autumn corner with burnt orange throw, apple cider mug, and leaves visible outside the window",
        "Calm neutral nursery corner with rocking chair, linen drape, and warm yellow lamp",
        "Hygge night corner with thick duvet, fairy lights canopy, and peppermint tea on bedside crate",
        "Rustic writing nook with antique typewriter, brass candle holder, and moody evening shadows",
        "Scandinavian breakfast corner with oatmeal bowl, linen napkin, and pale winter light",
        "Cozy loft corner with oversized floor pillow, tea tray, and rain tapping on skylight glass",
        "Aesthetic corner by bookshelf with knit pouf, lavender candle, and soft golden sunset",
        "Muted bohemian bedroom with textured quilt, pampas stems, and cappuccino on a rattan side table",
        "Calm tea ritual corner with clay teapot, incense, and soft overcast light through shoji-style screen",
        "Candlelit bathside corner with fluffy towel, herbal tea, and warm beige stone textures",
        "Cozy creative desk with watercolor palette, warm lamp glow, and coffee mug on linen mat",
        "Serene evening corner with low armchair, wool throw, and moonlight blending with warm indoor light",
        "Indoor rainy greenhouse nook with wooden bench, blanket, and steaming ginger tea",
        "Soft neutral living room with boucle chair, arched lamp, and peaceful minimalist decor",
        "Winter hygge corner with faux fur rug, glowing lantern, and hot cocoa topped with cinnamon",
        "Rustic farmhouse corner with checked blanket, candle trio, and latte beside a vintage clock",
        "Quiet dawn corner with sheer curtains, warm mug, journal page, and gentle pastel sky",
        "Cozy bedtime corner with stack of novels, chamomile tea, and soft amber wall sconce lighting",
    ]

    prompt = random.choice(prompts)
    print(f"[IMAGE] Selected prompt: {prompt}")

    print("[IMAGE] Loading Stable Diffusion model")
    pipe = StableDiffusionPipeline.from_pretrained(
        "OFA-Sys/small-stable-diffusion-v0-1",
        torch_dtype=torch.float32,
    )

    print("[IMAGE] Generating image")
    image = pipe(prompt, num_inference_steps=20).images[0]
    image = image.resize((1080, 1080), Image.LANCZOS).convert("RGBA")

    print("[IMAGE] Applying banner and text overlay")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([(0, 1000), (1080, 1080)], fill=(0, 0, 0, 160))
    image = Image.alpha_composite(image, overlay)

    draw = ImageDraw.Draw(image)
    text = "✨ @cozycorner4245 | Your Daily Comfort"
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 34)
    except OSError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 34)
        except OSError:
            print("[IMAGE] DejaVuSans.ttf not found, using default PIL font")
            font = ImageFont.load_default()

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = (1080 - text_width) // 2
    y = 1000 + (80 - text_height) // 2
    draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)

    image.convert("RGB").save(output_path, format="JPEG", quality=95)
    print(f"[IMAGE] Saved final image to {output_path}")


def publish_post(image_path):
    print("[POST] Starting Instagram publish flow")
    password = os.environ.get("IG_PASSWORD")
    if not password:
        raise RuntimeError("IG_PASSWORD environment variable is required.")

    username = "cozycorner4245"
    absolute_image_path = os.path.abspath(image_path)
    if not os.path.exists(absolute_image_path):
        raise RuntimeError(f"Image file not found: {absolute_image_path}")

    caption = """✨ Finding peace in cozy corners 🍵☁️
Your daily dose of warmth and calm.
#cozycorner #aesthetic #cozyvibes #slowliving #hygge #minimalist #cozyathome #warmth #selfcare #peacefulspace #coffeetime #readingtime #cozyhome #interiordesign #homesweethome #coziness #neutralaesthetic #softlife #tranquil #mindfulness"""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1080})
        page = context.new_page()

        try:
            print("[POST] Navigating to login page")
            page.goto("https://www.instagram.com/accounts/login/", wait_until="domcontentloaded", timeout=60000)
            page.wait_for_selector("input[name='username']", timeout=60000)

            print("[POST] Filling credentials")
            page.fill("input[name='username']", username)
            page.fill("input[name='password']", password)
            page.click("button[type='submit']")
            page.wait_for_load_state("networkidle", timeout=60000)

            print("[POST] Handling optional popups")
            for popup_text in ["Not Now", "Cancel"]:
                try:
                    page.get_by_role("button", name=popup_text).first.click(timeout=4000)
                    print(f"[POST] Dismissed popup: {popup_text}")
                except Exception:
                    print(f"[POST] Popup not found, skipping: {popup_text}")

            print("[POST] Opening new post composer")
            opened = _click_first_available(
                page,
                [
                    "[aria-label='New post']",
                    "[aria-label='New Post']",
                    "svg[aria-label='New post']",
                    "svg[aria-label='New Post']",
                    "a[href*='/create']",
                    "text=Create",
                    "button:has-text('Create')",
                ],
                timeout=10000,
            )
            if not opened:
                raise RuntimeError("Could not open the Instagram create post flow.")

            print("[POST] Uploading image")
            _upload_image(page, absolute_image_path)

            print("[POST] Advancing through editor screens")
            for index in range(3):
                clicked_next = _click_first_available(
                    page,
                    [
                        "button:has-text('Next')",
                        "div[role='button']:has-text('Next')",
                        "text=Next",
                    ],
                    timeout=7000,
                )
                if clicked_next:
                    print(f"[POST] Clicked Next ({index + 1})")
                else:
                    print("[POST] No additional Next button found, continuing")
                    break

            print("[POST] Filling caption")
            caption_selector = None
            for selector in [
                "textarea[aria-label='Write a caption…']",
                "textarea[aria-label='Write a caption...']",
                "textarea",
            ]:
                try:
                    page.wait_for_selector(selector, timeout=10000)
                    caption_selector = selector
                    break
                except Exception:
                    continue
            if not caption_selector:
                raise RuntimeError("Caption textarea was not found.")
            page.fill(caption_selector, caption)

            print("[POST] Sharing post")
            shared = _click_first_available(
                page,
                [
                    "button:has-text('Share')",
                    "div[role='button']:has-text('Share')",
                    "text=Share",
                ],
                timeout=10000,
            )
            if not shared:
                raise RuntimeError("Share button was not found.")

            print("[POST] Waiting for share confirmation")
            confirmation_found = False
            for selector in [
                "text=Your post has been shared",
                "text=Your post was shared",
                "text=Post shared",
            ]:
                try:
                    page.wait_for_selector(selector, timeout=60000)
                    confirmation_found = True
                    print(f"[POST] Confirmation detected with selector: {selector}")
                    break
                except Exception:
                    continue
            if not confirmation_found:
                raise RuntimeError("Post share confirmation was not detected within 60 seconds.")

            print("[POST] Publish flow completed successfully")
        finally:
            print("[POST] Closing browser context")
            context.close()
            browser.close()


if __name__ == "__main__":
    output = "generated_post.jpg"
    generate_ai_cozy_image(output)
    publish_post(output)
