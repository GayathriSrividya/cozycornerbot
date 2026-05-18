import os
import random
from datetime import datetime, timezone

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

# --- CONFIGURATION ---
USERNAME = os.environ.get("IG_USERNAME", "cozycorner4245")
PASSWORD = os.environ.get("IG_PASSWORD")
IMAGE_PATH = os.path.abspath("generated_post.jpg")
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
DEBUG_DIR = os.path.abspath("debug_artifacts")


def log_step(step_name, detail):
    print(f"[{step_name}] {detail}")


def ensure_debug_dir():
    os.makedirs(DEBUG_DIR, exist_ok=True)


def capture_debug_artifacts(page, stage, error=None):
    ensure_debug_dir()
    safe_stage = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in stage).strip("-") or "unknown-stage"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    prefix = os.path.join(DEBUG_DIR, f"{stamp}-{safe_stage}")
    screenshot_path = f"{prefix}.png"
    html_path = f"{prefix}.html"
    context_path = f"{prefix}.txt"

    try:
        page.screenshot(path=screenshot_path, full_page=True)
        log_step("DEBUG", f"Saved screenshot: {screenshot_path}")
    except Exception as exc:
        log_step("DEBUG", f"Failed to save screenshot at {stage}: {exc}")

    try:
        with open(html_path, "w", encoding="utf-8") as html_file:
            html_file.write(page.content())
        log_step("DEBUG", f"Saved page HTML: {html_path}")
    except Exception as exc:
        log_step("DEBUG", f"Failed to save HTML at {stage}: {exc}")

    try:
        with open(context_path, "w", encoding="utf-8") as context_file:
            context_file.write(f"timestamp_utc={datetime.now(timezone.utc).isoformat()}\n")
            context_file.write(f"stage={stage}\n")
            context_file.write(f"url={page.url}\n")
            context_file.write(f"title={page.title()}\n")
            if error:
                context_file.write(f"error={error}\n")
        log_step("DEBUG", f"Saved debug context: {context_path}")
    except Exception as exc:
        log_step("DEBUG", f"Failed to save context at {stage}: {exc}")


def click_first_available(page, selectors, timeout=5000, step_name="ACTION"):
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            locator.wait_for(state="visible", timeout=timeout)
            locator.click(timeout=timeout)
            log_step(step_name, f"Clicked selector: {selector}")
            return selector
        except Exception:
            continue
    return None


def dismiss_optional_popups(page):
    log_step("POPUPS", "Checking optional popups")
    popup_selectors = [
        "text=Not Now",
        "button:has-text('Not Now')",
        "button:has-text('Cancel')",
        "div[role='button']:has-text('Not Now')",
    ]
    for selector in popup_selectors:
        try:
            page.locator(selector).first.click(timeout=3000)
            log_step("POPUPS", f"Dismissed popup using selector: {selector}")
        except Exception:
            pass


def set_upload_file(page, image_path_to_upload):
    file_input = page.locator("input[type='file']").first

    try:
        file_input.set_files(image_path_to_upload, timeout=8000)
        log_step("UPLOAD", "Uploaded file directly via input[type='file']")
        return
    except Exception as direct_exc:
        log_step("UPLOAD", f"Direct file input not ready, falling back to chooser ({direct_exc})")

    try:
        with page.expect_file_chooser(timeout=10000) as chooser_info:
            clicked_selector = click_first_available(
                page,
                [
                    "text=Select From Computer",
                    "button:has-text('Select From Computer')",
                    "div[role='button']:has-text('Select From Computer')",
                    "text=Select from computer",
                ],
                timeout=7000,
                step_name="UPLOAD",
            )
            if not clicked_selector:
                raise RuntimeError("Could not find a 'Select From Computer' trigger")
        chooser_info.value.set_files(image_path_to_upload)
        log_step("UPLOAD", "Uploaded file via file chooser")
    except Exception as chooser_exc:
        raise RuntimeError(
            f"Failed to upload image file via input and chooser fallbacks: {chooser_exc}"
        ) from chooser_exc


def wait_for_publish_confirmation(page):
    confirmation_selectors = [
        "text=Your post was shared",
        "text=Post shared",
        "text=Shared",
    ]
    for selector in confirmation_selectors:
        try:
            page.wait_for_selector(selector, timeout=30000)
            log_step("CONFIRM", f"Publish confirmation found via selector: {selector}")
            return
        except Exception:
            continue
    raise RuntimeError(
        "Publish confirmation not found. Tried selectors: "
        + ", ".join(confirmation_selectors)
    )


def generate_ai_cozy_image(output_path):
    """Generates a theme-specific image using 1 of 50 randomized cozy prompts."""
    log_step("IMAGE", "Initializing AI Image Generator")
    cozy_prompts = [
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
        "An aesthetic bedroom corner, a bed layered with heavy linen duvets in eucalyptus green and soft white, string lights headboard glow",
        "A close-up of a small green succulent growing out of a tiny handmade clay pot, sitting on a textured stone coaster, bright crisp lighting",
        "A wicker basket overflowing with rolled up waffle-knit towels and soft cotton blankets, sitting next to a tall indoor fiddle-leaf fig plant",
        "A low floor mattress arrangement with crisp white sheets and a chunky knit chunky blanket, soft boho chic aesthetic, morning light",
        "A rustic stone fireplace mantle decorated with small jars of dried baby's breath flowers and a single glowing amber candle",
        "A close-up texture shot of hand-woven beige macrame fabric hanging against a clean white textured plaster wall, soft side lighting",
        "A warm greenhouse corner, small clay plant pots arranged on wooden steps, a watering can, soft humid sunlight filtering through glass",
        "A bedside scene, a single white rose in a small glass bud vase, a linen book laying open next to it face down, soft focus",
        "A cluster of different sized aesthetic candles burning at different heights on a mirror tray, creating a warm dancing golden glow",
        "A simple white porcelain bowl filled with fresh ripe strawberries sitting on an unbleached linen tablecloth, bright airy morning vibe",
    ]

    selected_prompt = random.choice(cozy_prompts)
    log_step("IMAGE", f"AI Prompt Selected: {selected_prompt}")

    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
    )

    log_step("IMAGE", "AI is painting your cozy scene")
    ai_image = pipe(selected_prompt, num_inference_steps=20, height=512, width=512).images[0]
    ai_image = ai_image.resize((1080, 1080), Image.Resampling.LANCZOS)

    ai_image = ai_image.convert("RGBA")
    overlay = Image.new("RGBA", ai_image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([(0, 1020), (1080, 1080)], fill=(0, 0, 0, 80))
    ai_image = Image.alpha_composite(ai_image, overlay)

    draw = ImageDraw.Draw(ai_image)
    try:
        font = ImageFont.truetype(FONT_PATH, 35)
    except IOError:
        font = ImageFont.load_default()

    draw.text((40, 1035), f"✨ @{USERNAME} | Your Daily Comfort", fill="#FFFFFF", font=font)

    ai_image.convert("RGB").save(output_path, quality=95)
    log_step("IMAGE", f"AI image saved successfully to {output_path}")


def publish_post(image_path_to_upload):
    """Uses Playwright automation to publish the AI generated image."""
    caption = (
        "✨ Pull up a chair, grab a warm drink, and settle in. Welcome to your cozy corner. "
        f"\n.\n.\n#cozycorner #cozyvibes #aesthetic #lofi #mentalpeace #warmth #{datetime.now().strftime('%Y')}"
    )

    if not PASSWORD:
        raise RuntimeError("IG_PASSWORD environment variable is missing in GitHub Secrets.")

    if not os.path.exists(image_path_to_upload):
        raise RuntimeError(f"Image to upload was not found: {image_path_to_upload}")

    ensure_debug_dir()

    with sync_playwright() as p:
        device = p.devices["iPhone 12 Pro"]
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(**device)
        page = context.new_page()
        page.set_default_timeout(15000)

        try:
            log_step("LOGIN", "Navigating to Instagram login page")
            page.goto("https://www.instagram.com/accounts/login/", wait_until="domcontentloaded")

            log_step("LOGIN", "Submitting login credentials")
            page.fill("input[name='username']", USERNAME)
            page.fill("input[name='password']", PASSWORD)
            page.locator("input[name='password']").press("Enter")
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(3000)
            log_step("LOGIN", "Login flow submitted")

            dismiss_optional_popups(page)

            log_step("COMPOSER", "Opening post composer")
            composer_selector = click_first_available(
                page,
                [
                    "svg[aria-label='New Post']",
                    "svg[aria-label='New post']",
                    "button:has(svg[aria-label='New Post'])",
                    "button:has(svg[aria-label='New post'])",
                    "a[aria-label='New Post']",
                    "a[aria-label='New post']",
                    "[aria-label='New Post']",
                    "[aria-label='New post']",
                    "a[href='/create/select/']",
                    "a[href*='/create']",
                    "div[role='button']:has-text('Create')",
                    "button:has-text('Create')",
                    "[role='menuitem']:has-text('Post')",
                    "text=Create",
                ],
                timeout=8000,
                step_name="COMPOSER",
            )
            if not composer_selector:
                raise RuntimeError("Could not open Instagram post composer using available selectors.")

            log_step("UPLOAD", "Selecting file for upload")
            set_upload_file(page, image_path_to_upload)
            page.wait_for_timeout(4000)

            log_step("NEXT", "Advancing through composer screens")
            next_selectors = [
                "div[role='button']:has-text('Next')",
                "button:has-text('Next')",
                "text=Next",
            ]
            first_next = click_first_available(page, next_selectors, timeout=12000, step_name="NEXT")
            if not first_next:
                raise RuntimeError("Failed to click first 'Next' button in composer flow.")

            second_next = click_first_available(page, next_selectors, timeout=6000, step_name="NEXT")
            if second_next:
                log_step("NEXT", "Clicked optional second 'Next' button")

            log_step("CAPTION", "Filling post caption")
            caption_target = page.locator("textarea[aria-label='Write a caption...']").first
            caption_target.wait_for(state="visible", timeout=10000)
            caption_target.fill(caption)

            log_step("SHARE", "Submitting post")
            share_selector = click_first_available(
                page,
                [
                    "div[role='button']:has-text('Share')",
                    "button:has-text('Share')",
                    "text=Share",
                ],
                timeout=12000,
                step_name="SHARE",
            )
            if not share_selector:
                raise RuntimeError("Failed to click 'Share' button.")

            log_step("CONFIRM", "Waiting for upload confirmation")
            wait_for_publish_confirmation(page)
            log_step("DONE", "Success! A beautiful new cozy view has been uploaded.")
        except Exception as exc:
            capture_debug_artifacts(page, "publish-failure", error=str(exc))
            raise RuntimeError(
                f"Instagram publish flow failed: {exc}. Debug artifacts saved to {DEBUG_DIR}"
            ) from exc
        finally:
            browser.close()


if __name__ == "__main__":
    if os.path.exists(IMAGE_PATH):
        os.remove(IMAGE_PATH)

    generate_ai_cozy_image(IMAGE_PATH)
    if os.path.exists(IMAGE_PATH):
        publish_post(IMAGE_PATH)
