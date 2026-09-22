"""Integration check of the running web UI; artifacts stay under runs/."""
from pathlib import Path
import json
from playwright.sync_api import sync_playwright

out = Path(__file__).resolve().parents[1] / 'runs/web_ui_260922'
out.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='/usr/bin/google-chrome', headless=True, args=['--no-sandbox'])
    page = browser.new_page(viewport={'width': 1440, 'height': 1100})
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto('http://127.0.0.1:8765')
    page.locator('.frame').first.wait_for()
    page.locator('.image-stage img').first.wait_for()
    page.wait_for_function('document.querySelector(".image-stage img").naturalWidth > 0')
    catalog=page.request.get('http://127.0.0.1:8765/api/catalog').json()
    assert catalog['stats']['frames']==1713
    current=page.locator('.dataset-heading h2').inner_text()
    expected=page.request.get('http://127.0.0.1:8765/api/experiments/'+current).json()['cases']
    assert page.locator('.frame').count()==sum(len(c['frames']) for c in expected)
    page.screenshot(path=str(out/'desktop.png'),full_page=False)
    page.locator('button.image-stage').first.click()
    assert page.locator('dialog').is_visible()
    page.keyboard.press('Escape')
    assert not page.locator('dialog').is_visible()
    page.get_by_role('button',name='重建对比',exact=True).click()
    assert page.locator('.image-grid').first.locator('.image-card').count()==2
    page.get_by_label('查看内容').select_option('scanner_preview')
    assert page.locator('.image-grid').first.locator('.image-card').count()==1
    page.reload()
    page.locator('.frame').first.wait_for()
    assert page.get_by_label('查看内容').input_value()=='scanner_preview'
    page.get_by_label('搜索实验').fill('NO_MATCH_12345')
    assert page.get_by_text('未找到匹配实验').is_visible()
    page.get_by_label('搜索实验').fill('')
    # Traverse a real experiment that contains unavailable or partial data.
    target=next(c['experiment'] for c in catalog['cases'] if c['status']=='raw_unavailable')
    page.locator('.experiment').filter(has=page.locator('strong',has_text=target)).first.click()
    page.wait_for_function('(name)=>document.querySelector(".dataset-heading h2").textContent===name',arg=target)
    page.locator('.no-frames').first.wait_for()
    page.get_by_role('button',name='四图对比',exact=True).click()
    page.set_viewport_size({'width':390,'height':844})
    page.screenshot(path=str(out/'mobile.png'),full_page=False)
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    assert not errors, errors
    (out/'checks.json').write_text(json.dumps({'passed':True,'checks':['real frame coverage','zoom and Escape','view modes','URL persistence','search empty state','experiment switch','unavailable scan','mobile overflow'],'browser_errors':errors},ensure_ascii=False,indent=2))
    browser.close()
print('Browser checks passed')
