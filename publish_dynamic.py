from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

def click_first_available(page, selectors, timeout=5000):
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            locator.wait_for(state="visible", timeout=timeout)
            locator.click(timeout=timeout)
            print(f"Clicked selector: {selector}")
            return True
        except Exception:
            continue
    return False

def publish_post(image_path_to_upload):
    CAPTION = f"✨ Pull up a chair, grab a warm drink, and settle in. Welcome to your cozy corner. \n.\n.\n#cozycorner #cozyvibes #aesthetic #lofi #mentalpeace #warmth #{datetime.now().strftime('%Y')}"

    if not PASSWORD:
        print("Error: IG_PASSWORD environment variable is missing in GitHub Secrets.")
        return

    with sync_playwright() as p:
        device = p.devices["iPhone 12 Pro"]
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(**device)
        page = context.new_page()

        print("Navigating to Instagram...")
        page.goto("https://www.instagram.com/accounts/login/", wait_until="domcontentloaded")

        print("Logging in...")
        page.fill("input[name='username']", USERNAME)
        page.fill("input[name='password']", PASSWORD)
        page.locator("input[name='password']").press("Enter")

        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(3000)

        for selector in ["text=Not Now", "button:has-text('Not Now')", "button:has-text('Cancel')"]:
            try:
                page.locator(selector).first.click(timeout=3000)
            except Exception:
                pass

        print("Opening post composer...")
        opened = click_first_available(page, [
            "svg[aria-label='New Post']",
            "svg[aria-label='New post']",
            "a[aria-label='New Post']",
            "button:has(svg[aria-label='New Post'])",
            "button:has(svg[aria-label='New post'])",
            "[role='menuitem']:has-text('Post')",
            "text=Create",
        ], timeout=5000)

        file_input = page.locator("input[type='file']").first

        if not opened:
            print("Could not find New Post button, trying file input directly...")
        try:
            file_input.set_files(image_path_to_upload, timeout=5000)
        except Exception:
            try:
                with page.expect_file_chooser(timeout=5000) as file_chooser_info:
                    clicked = click_first_available(page, [
                        "text=Select From Computer",
                        "button:has-text('Select From Computer')",
                        "div[role='button']:has-text('Select From Computer')",
                    ], timeout=5000)
                    if not clicked:
                        raise RuntimeError("Could not find upload trigger")
                file_chooser_info.value.set_files(image_path_to_upload)
            except Exception as exc:
                page.screenshot(path="instagram-upload-failure.png", full_page=True)
                raise RuntimeError(f"Failed to open upload flow: {exc}")

        page.wait_for_timeout(5000)

        click_first_available(page, [
            "div[role='button']:has-text('Next')",
            "button:has-text('Next')",
        ], timeout=10000)

        page.locator("textarea[aria-label='Write a caption...']").fill(CAPTION)

        click_first_available(page, [
            "div[role='button']:has-text('Share')",
            "button:has-text('Share')",
        ], timeout=10000)

        print("Waiting for upload confirmation...")
        page.wait_for_selector("text=Your post was shared", timeout=30000)

        print("Success! A beautiful new cozy view has been uploaded.")
        browser.close()
