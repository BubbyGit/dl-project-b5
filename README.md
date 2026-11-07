# B5. Derandomized flip на airbench94

проверяю утверждение из статьи Keller Jordan "94% on CIFAR-10 in 3.29 Seconds on a Single GPU" (https://arxiv.org/abs/2404.00498): alternating flip даёт точность выше, чем обычный random flip, везде где горизонтальные отражения вообще помогают. оригинальный код https://github.com/KellerJordan/cifar10-airbench, беру `legacy/airbench94.py` с коммита 4c1b6d1e3889b037efadcfd5c0ea65b246592362. он лежит в `original/` без изменений вместе с лицензией MIT автора, совпадение с upstream можно проверить, обе последние команды должны вывести 74832178a5ad1231fa37e82be111b24339644fa7
```
git clone https://github.com/KellerJordan/cifar10-airbench.git
git -C cifar10-airbench checkout 4c1b6d1e3889b037efadcfd5c0ea65b246592362
git -C cifar10-airbench rev-parse HEAD:legacy/airbench94.py
git hash-object original/airbench94.py
```

нужна nvidia gpu от turing и новее (sm_75+) и хотя бы 4 гб видеопамяти, на rtx 5070 обучение занимает около 3.3 гб, весь датасет лежит на gpu. python 3.11+, у меня 3.12.9. torch 2.14.1 и torchvision 0.29.1 со сборкой cuda 13.0, она работает и на t4, и на rtx 50xx, остальное в `requirements.txt`
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
на windows вместо source `.venv\Scripts\activate`. в colab torch уже стоит, хватит `pip install pyyaml scipy`. cifar-10 скачивается сам в `cifar10/` при первом запуске, но cs.toronto.edu бывает очень медленным, тогда тот же архив можно заранее взять с зеркала, md5 должен быть c58f30108f718f92721af3b95e74349a
```
mkdir -p cifar10
curl -L -o cifar10/cifar-10-python.tar.gz https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python/resolve/main/cifar-10-python.tar.gz
```

все команды из корня репозитория. оригинал как есть (25 обучений по 9.9 эпох с alternating flip, точность без tta и с tta) запускается через обёртку, которая просто сохраняет stdout, время, log.pt и окружение в `results/checkpoint1/original/`. без обёртки то же самое это `python -u original/airbench94.py`
```
python -m flipbench.run_original
```

тесты проверяют, что код экспериментов в режиме alternating выдаёт те же батчи, что и оригинал, и что режимы flip работают как надо
```
python -m pytest
```

эксперименты задаются конфигами в `configs/`, каждый прогон дописывается строкой в `results/<name>/runs.jsonl`. если серию прервать, повторный запуск продолжит с того же места. `--seed-count` берёт только первые n сидов. потом summarize строит таблицы со средним, std, 95% доверительным интервалом и парными разностями режимов по сидам, рядом значения из статьи, а budget считает время и нужное число сидов по пилоту
```
python -m flipbench.sweep configs/table6.yaml
python -m flipbench.sweep configs/table6.yaml --seed-count 10
python -m flipbench.summarize results/table6
python -m flipbench.budget --out results/checkpoint1/budget.txt
```

режимов flip три. none без отражений. random каждую эпоху отражает каждую картинку с вероятностью 0.5. alternating как в оригинале, в эпоху 0 картинка отражается с вероятностью 0.5, а дальше её ориентация меняется каждую эпоху. всё остальное (архитектура, оптимизатор, lr, batch size, оценка) берётся из airbench94.py, меняются только режим flip, число эпох и translate. seed задаёт отдельные генераторы для инициализации, порядка, сдвигов и отражений, поэтому при одном seed у всех режимов одинаковые инициализация, порядок и сдвиги, и сравнивать их можно попарно

что получилось к чекпоинту 1. оригинал запустился без изменений кода, поменялось только окружение (новый torch под rtx 5070 и архив cifar с зеркала, подробно в `DECISIONS.md`). 25 обучений по 9.9 эпох на rtx 5070 12 гб, windows 11, python 3.12.9, torch 2.14.1+cu130, torchvision 0.29.1+cu130, cuda 13.0, cudnn 9.24, драйвер 616.64. точность без tta 93.26 ± 0.14 % (среднее ± std по 25 обучениям, от 92.99 до 93.50), с tta 94.02 ± 0.15 % (от 93.76 до 94.29). в статье 93.2 % без tta и 94.01 % с tta по 1000 обучениям, то есть сошлось. одно обучение по метрике самого оригинала total_time_seconds (whitening, обучение и tta без промежуточных оценок) занимает 25.9 ± 0.3 с, первое 27.4 с из-за автотюна cudnn, у автора на a100 было 3.83 с. вся команда целиком шла 672.5 с, то есть около 27 с на обучение вместе с оценкой после каждой эпохи. архив cifar с зеркала скачался за 10 с. всё лежит в `results/checkpoint1/original/`: `stdout.txt` это полный лог, `run.json` метрики каждого обучения и окружение, `log.pt` сохранил сам оригинал, в `environment/` версии, pip freeze и nvidia-smi

дальше по плану после принятия pre-registration. `configs/table6.yaml` это ядро, Table 6 статьи без cutout: 10 и 20 эпох, три режима, по 50 сидов, точность без tta и с tta. `configs/translate.yaml` это расширение: меняется ли выигрыш alternating flip от силы translate (0, 2, 4) на 10 эпохах. решения и почему они такие лежат в `DECISIONS.md`
