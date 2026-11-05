"""Открытые проверки из ноутбуков ДЗ1.

`CHECKS` — список проверок, `run(target)` прогоняет их по объекту с
реализацией и возвращает список падений. Проверки не скрыты: при приёме
прогоняются ровно они.
"""
import math

import torch
import torch.nn.functional as F

CHECKS = []


def check(title, hint):
    def wrap(fn):
        CHECKS.append(dict(name=fn.__name__, title=title, hint=hint, fn=fn))
        return fn
    return wrap


def close(a, b, tol=1e-5):
    return abs(float(a) - float(b)) <= tol


def cfg(m, **kw):
    base = dict(vocab_size=32, context_size=16, d_model=32, n_heads=4, n_layers=2, dropout=0.0)
    base.update(kw)
    return m.GPTConfig(**base)


# --- позиционное кодирование --------------------------------------------------

@check('позиционное кодирование совпадает с формулой',
       'чётные каналы должны быть синусом, нечётные — косинусом одной и той же частоты')
def positional_encoding_matches_formula(m):
    length, d_model = 8, 16
    pe = m.sinusoidal_positional_encoding(length, d_model)
    assert pe.shape == (length, d_model), f'форма {tuple(pe.shape)}, ожидалась {(length, d_model)}'
    for pos in range(length):
        for i in range(0, d_model, 2):
            w = math.exp(-math.log(10000.0) * i / d_model)
            assert close(pe[pos, i], math.sin(pos * w)), f'канал {i} на позиции {pos}'
            assert close(pe[pos, i + 1], math.cos(pos * w)), f'канал {i + 1} на позиции {pos}'


@check('нулевая позиция закодирована нулевой позицией',
       'проверьте, с какого числа начинается отсчёт позиций')
def positional_encoding_starts_at_zero_position(m):
    pe = m.sinusoidal_positional_encoding(4, 8)
    assert torch.allclose(pe[0, 0::2], torch.zeros(4), atol=1e-6), 'синусы на позиции 0 не нулевые'
    assert torch.allclose(pe[0, 1::2], torch.ones(4), atol=1e-6), 'косинусы на позиции 0 не единичные'


# --- нормализация -------------------------------------------------------------

@check('LayerNorm нормирует каждый токен отдельно',
       'посмотрите, по какой оси считаются среднее и дисперсия')
def layernorm_normalizes_each_token_separately(m):
    norm = m.LayerNorm(16)
    x = torch.randn(4, 6, 16) * 5 + 3
    x[0] *= 100
    y = norm(x)
    assert torch.allclose(y.mean(dim=-1), torch.zeros(4, 6), atol=1e-4), 'среднее по признакам не ноль'
    assert torch.allclose(y.var(dim=-1, unbiased=False), torch.ones(4, 6), atol=1e-3), \
        'дисперсия по признакам не единица'


@check('нормировка одного примера не зависит от остальных в батче',
       'если статистики считаются по батчу, получается batch norm, а не layer norm')
def layernorm_is_independent_of_other_batch_elements(m):
    norm = m.LayerNorm(16)
    a = torch.randn(1, 3, 16)
    b = torch.randn(1, 3, 16) * 50
    alone = norm(a)
    together = norm(torch.cat([a, b], dim=0))[:1]
    assert torch.allclose(alone, together, atol=1e-5), 'соседний пример батча изменил результат'


# --- головы -------------------------------------------------------------------

@check('merge_heads в точности обращает split_heads',
       'одна из двух функций переставляет оси не так, как другая')
def split_merge_roundtrip(m):
    x = torch.randn(2, 5, 32)
    back = m.merge_heads(m.split_heads(x, 4))
    assert back.shape == x.shape, f'форма после сборки {tuple(back.shape)}, ожидалась {tuple(x.shape)}'
    assert torch.allclose(back, x, atol=1e-6), 'содержимое после сборки не совпало с исходным'


@check('голова получает свой кусок каналов',
       'нужную форму дают два способа; только один кладёт в голову h её собственные каналы')
def split_heads_shape_and_layout(m):
    x = torch.randn(2, 5, 32)
    heads = m.split_heads(x, 4)
    assert heads.shape == (2, 4, 5, 8), f'форма {tuple(heads.shape)}, ожидалась (2, 4, 5, 8)'
    for h in range(4):
        assert torch.allclose(heads[:, h], x[..., h * 8:(h + 1) * 8], atol=1e-6), \
            f'в голове {h} лежат не её каналы'


# --- внимание -----------------------------------------------------------------

@check('внимание совпадает с эталонной реализацией PyTorch',
       'расходятся масштаб логитов, порядок Q и K либо ось softmax')
def attention_matches_reference_implementation(m):
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 6, 8) for _ in range(3))
    out, _ = m.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(out, F.scaled_dot_product_attention(q, k, v), atol=1e-5)


@check('веса внимания суммируются к единице по ключам',
       'каждая строка матрицы внимания обязана быть распределением')
def attention_weights_sum_to_one_over_keys(m):
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 6, 8) for _ in range(3))
    _, w = m.scaled_dot_product_attention(q, k, v)
    assert w.shape == (2, 4, 6, 6), f'форма весов {tuple(w.shape)}, ожидалась (2, 4, 6, 6)'
    assert torch.allclose(w.sum(dim=-1), torch.ones(2, 4, 6), atol=1e-5)


@check('масштаб логитов зависит от корня из d_k',
       'при d_k = 64 деление на d_k вместо корня даёт заметно другие веса')
def attention_scale_depends_on_sqrt_d_k(m):
    torch.manual_seed(1)
    q, k, v = (torch.randn(1, 1, 12, 64) for _ in range(3))
    _, w = m.scaled_dot_product_attention(q, k, v)
    expected = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(64), dim=-1)
    assert torch.allclose(w, expected, atol=1e-5)


@check('причинная маска обнуляет запрещённые позиции',
       'важно, на каком шаге применяется маска и какое значение подставляется')
def attention_respects_causal_mask(m):
    torch.manual_seed(2)
    q, k, v = (torch.randn(1, 1, 5, 8) for _ in range(3))
    mask = torch.tril(torch.ones(5, 5)).view(1, 1, 5, 5)
    _, w = m.scaled_dot_product_attention(q, k, v, mask)
    upper = w[0, 0].triu(diagonal=1)
    assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-7), \
        'будущие позиции получили ненулевой вес'
    assert torch.allclose(w.sum(dim=-1), torch.ones(1, 1, 5), atol=1e-5), \
        'после маскирования веса больше не суммируются к единице'


@check('модель не подглядывает в будущее',
       'меняем будущие токены — предсказания на прошлых позициях обязаны не измениться')
def no_future_leak_in_model(m):
    torch.manual_seed(2)
    model = m.MiniGPT(cfg(m)).eval()
    a = torch.randint(0, 32, (1, 12))
    b = a.clone()
    b[0, 7:] = torch.randint(0, 32, (5,))
    with torch.no_grad():
        la, _, _ = model(a)
        lb, _, _ = model(b)
    assert torch.allclose(la[:, :7], lb[:, :7], atol=1e-5)


# --- полносвязный подслой и блок ----------------------------------------------

@check('FFN нелинеен',
       'две проекции подряд без нелинейности схлопываются в одно линейное преобразование')
def ffn_is_nonlinear(m):
    torch.manual_seed(3)
    ffn = m.FeedForward(cfg(m))
    a, b = torch.randn(1, 1, 32), torch.randn(1, 1, 32)
    gap = (ffn(a + b) - ffn(a) - ffn(b) + ffn(torch.zeros_like(a))).abs().max()
    assert gap > 1e-3, 'подслой ведёт себя как линейный'


@check('блок сохраняет остаточный путь',
       'обнуляем выходы подслоёв — блок обязан вернуть вход без изменений')
def block_keeps_residual_path(m):
    torch.manual_seed(4)
    block = m.TransformerBlock(cfg(m))
    with torch.no_grad():
        block.attn.w_o.weight.zero_()
        block.ffn.down.weight.zero_()
        block.ffn.down.bias.zero_()
    x = torch.randn(2, 6, 32)
    out, _ = block(x)
    assert torch.allclose(out, x, atol=1e-5)


# --- модель и генерация -------------------------------------------------------

@check('модель отдаёт логиты, потери и карты внимания нужных форм',
       'проверьте, что forward возвращает кортеж (логиты, потери, список карт внимания)')
def model_shapes_and_loss(m):
    torch.manual_seed(5)
    model = m.MiniGPT(cfg(m))
    idx = torch.randint(0, 32, (3, 10))
    logits, loss, attentions = model(idx, targets=idx)
    assert logits.shape == (3, 10, 32), f'логиты {tuple(logits.shape)}, ожидалось (3, 10, 32)'
    assert loss.item() > 0, 'потери не положительны'
    assert len(attentions) == 2, f'карт внимания {len(attentions)}, ожидалось 2 (по слою)'
    assert attentions[0].shape == (3, 4, 10, 10), f'карта {tuple(attentions[0].shape)}'


@check('при top_k=1 генерация совпадает с argmax',
       'посмотрите, от какого конца отсортированных логитов берётся порог')
def generate_respects_top_k_one(m):
    torch.manual_seed(5)
    model = m.MiniGPT(cfg(m)).eval()
    idx = torch.randint(0, 32, (1, 4))
    out1 = m.generate(model, idx, max_new_tokens=3, top_k=1)
    out2 = m.generate(model, idx, max_new_tokens=3, top_k=1)
    assert torch.equal(out1, out2), 'два одинаковых вызова дали разное'
    with torch.no_grad():
        logits, _, _ = model(idx)
    assert out1[0, 4].item() == logits[0, -1].argmax().item(), 'выбран не самый вероятный токен'


@check('генерация переводит модель в режим инференса',
       'что делает dropout, если модель осталась в режиме обучения')
def generate_switches_model_to_eval(m):
    torch.manual_seed(6)
    model = m.MiniGPT(cfg(m, dropout=0.5)).train()
    idx = torch.randint(0, 32, (1, 4))
    a = m.generate(model, idx, max_new_tokens=4, top_k=1)
    b = m.generate(model, idx, max_new_tokens=4, top_k=1)
    assert torch.equal(a, b), 'генерация нестабильна от вызова к вызову'


@check('модель обучается: потери падают на одной последовательности',
       'интеграционная проверка; если падает только она, ищите ошибку в градиентном пути')
def model_can_overfit_tiny_sequence(m):
    torch.manual_seed(7)
    model = m.MiniGPT(cfg(m, n_layers=1))
    idx = torch.randint(0, 32, (2, 12))
    x, y = idx[:, :-1], idx[:, 1:]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    _, first, _ = model(x, targets=y)
    loss = first
    for _ in range(120):
        opt.zero_grad()
        _, loss, _ = model(x, targets=y)
        loss.backward()
        opt.step()
    assert loss.item() < first.item() * 0.5, \
        f'потери почти не изменились: {first.item():.3f} -> {loss.item():.3f}'


# --- запуск -------------------------------------------------------------------

def run(target, verbose=True):
    """Прогоняет все проверки по объекту с реализацией. Возвращает список падений."""
    failures = []
    for c in CHECKS:
        try:
            c['fn'](target)
        except NotImplementedError as exc:
            failures.append((c, f'не реализовано: {exc}'))
            mark, note = 'ПУСТО ', 'функция ещё не дописана'
        except AssertionError as exc:
            failures.append((c, str(exc) or 'проверка не прошла'))
            mark, note = 'ОШИБКА', (str(exc) or '') + f'\n         подсказка: {c["hint"]}'
        except Exception as exc:                       # noqa: BLE001
            failures.append((c, f'{type(exc).__name__}: {exc}'))
            mark, note = 'СБОЙ  ', f'{type(exc).__name__}: {exc}'
        else:
            mark, note = 'ok    ', ''
        if verbose:
            print(f'{mark} {c["title"]}')
            if note:
                print(f'         {note}')
    if verbose:
        total, bad = len(CHECKS), len(failures)
        print(f'\nпройдено {total - bad} из {total}')
        if not bad:
            print('Все проверки зелёные.')
    return failures