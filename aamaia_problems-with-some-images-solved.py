import pandas as pd

test_image_ids = ['b068abf33d16f3d9', 'af143fabbe5dbeca']
train_image_ids = ['0700490d4585a87f',
 '9475fe0280018f01',
 '97f18d062160ec3c',
 '4eea5ffdb5bd7200',
 '613ab30b92d5c7dc',
 '5263f52ca33c02f9',
 '71c4a866d6444142',
 '8ba070a030c95c36',
 '2e0d6a9666f53223',
 '23e791baad220927',
 '0701f8e60dbe7b73',
 '3107ff12417cbcc9',
 '1c09e9bb84e74b62',
 '0cfca20077419643',
 '4e2fdcce11faf0af',
 '2d9daae234601cff',
 'c85a0f3f7df4e8bb',
 '2984f74f76989354',
 'f378e50eb115c9d0',
 'a217e706dbbb0ffe',
 'b659d849421ad458',
 '372d8bca352b097e',
 'c3c0126b8090ffb7',
 'a2a0a9a3d4a2b440',
 '217daee32e74777d',
 '89f4669a549d6174',
 'c89128dadf2c274c',
 '00620277faf781a3',
 'c7868d91e91d8fcf',
 '2c1b70355b184cb2',
 '639c51c6c4841051',
 '953368f296201311',
 'ab3749ae485b1d8b',
 'a26a25705dc3ec1a',
 '05c7c7732606a379',
 '26f8434534aaf15e',
 '7cf4ffccd6b85447',
 '2dd1ea0e0e18a80e',
 '66a7f86e62f6fa33',
 '552b064f406d3b27',
 '210cf3b7691aff28',
 '3ff408917773bd8f',
 '5f1c9e34ef8c78e8',
 'a11ac2566444ba3e',
 'f3f7959792b94e5b',
 '0e7da474b4ccdbe3',
 '315e38723c9baaf3',
 '0066e19f7c52bf42',
 '4a671c179f34dcf2',
 'd081dc98200f8740',
 'ec382f209c77726c',
 '6c2e85cabb6ed4c8',
 'ccda8da994ea059f',
 '2e00edea66336430',
 '871ba76309bb84c6',
 'fa1fd0e221c118d9',
 '9b34ff486a6a07a4',
 '386c849147188612',
 'cb19063dc4531b72',
 '97a1d69d7545ca5a',
 '1ef755d7cbc434f4',
 '79a6d94885442064',
 '26fbfc9d71edc037',
 '58cd6a1df5da198a',
 '8fb0160bf8ddc4e5',
 'b4b0c7afe895f7a2',
 '816e8d7a406b88e2',
 '7ccc54e7744ad545',
 '4b312ead187f08b4',
 '7faaf7d7eb51cda6',
 'ad0251c855caaf36',
 '02178f532d535d6a',
 'f6a3b282130b8a7b',
 '97550d7a43d8fc92',
 'ed51f2239eae7f51',
 '70738d10ad66c4f2',
 '62a233469a2c3ef2',
 '38c28d4887b1f19f',
 '35076f40cfe80145',
 'cae7b1d22b6b6b7e',
 'f510b55137115e70',
 '7135abe5de1398bb',
 '75001b900d7bb78e',
 '1f4fa3508fccc456',
 '10f88fa39bc76e52',
 '20436948bb02ad90',
 'be2c5ff8684bb601',
 'd2b081351d052e0e',
 '6df68641310630f9',
 '14e58a226e188dc9',
 '77228c57432b9b1a',
 'caeb28c1cf885087',
 '8c87eaf82e994d76',
 'd2b8c3ba91f15ed6',
 'ef190af66763360f',
 '9692a01bbd645a28',
 '41ecc723bc1271d1',
 'd0c2411c5ccc7b9d',
 '5a0a624917bec0b0',
 'b6172f96070d5f08',
 'd223ee1c2cc5029e',
 '947809039e590562',
 '7afb3fd89c1d362b',
 'd0a695116045677f',
 'f3a62058147fe0dc',
 'b1c605e5bce9c25b',
 '199f481eafa193cd',
 'cd00e25ed66c4942',
 '3c848a675c915c73',
 '696657466ccd2752',
 'b893a88a3f8fd478',
 '3cc850bc3b34b6e6',
 '98f5e4abf2ff8271',
 '6fcbb898a3d2dcf2',
 'bba5ffeabbfae499',
 '887df09cbbeae864',
 'f5a6b16926141ee2',
 'bee24647ea8c21ad',
 'dbaaecf3db237c14',
 'b597e3cab538d83a',
 'ddf2c5f529ac7d27',
 'a23e57d5918939f5',
 '3d82461a27d80190',
 '25c0eef66db8b5b6',
 '1d9eb416726c5b23',
 '66e0345c98207378',
 'c6d4cb41c41361bd',
 '53837d51541a299b',
 '0837bad5eaf96263',
 'acf26d4316c21efe',
 '6fcc552a30144215',
 '8920dab8fd8444dc',
 '736b960ba11c275f',
 '28f9b0bebc9be96b',
 'a25939dc7332cbaa',
 '88b0e6f627a9bb16',
 '5bbb812428292189',
 'b8bcd0f82429ff31',
 '92cb5b75b76e10c0',
 'f301eb7a61e261f5',
 '421683eed5c91a5d',
 'cfd0dfc3c920168a',
 '3c767dcfd0a0d4d4',
 'c5ad250a422e592b',
 '552dd5ef267453d3',
 '3bf7f8a550c57a9b',
 '56d0b0835e367d6a',
 '381d7ed85efea21b',
 'c85bcc915bdfac63',
 '4ecc9365a9633076',
 '1f789dc971a0d27f',
 'dbda0dd6393f4877',
 'd732c52e992546f2',
 '24b3dee8f9bf0e42',
 '14fa5bce9f7354dd',
 '6e3bac43fd8e7681',
 'ea165808c3559bcd',
 '6d5ec44438ed9aff',
 '62a2098362d14520',
 '1ab4de16bca1c054',
 'faa6b2e240e2ac2c',
 'bf103bc41998c6d5',
 '3c2972cd117bd348',
 'cfe0450405d9f99e',
 '5051fddb09273327',
 '49044341e7bda5af',
 '8b8d9f5e528e41f5',
 '105b1dd080d44578',
 '999db14f7a1c3e93',
 '901ac1ab431decd6',
 'c8c653400e5f071c',
 '263e792763f80403',
 'e824d20ede4224f2',
 '92c00bd322cce506',
 '88496c017a31cfdb',
 '1751c2953ee4379f',
 '871e12511988adcf',
 '620a5539bb3d222f',
 '8e6dadca4eeb8b16',
 '6f82249401ee7d62',
 'e4c70d7a53e8cb48',
 'e570dc12933455a0',
 '777d5e4b6dcfa3bb',
 '4421c7a49c333061',
 '98117b66dc243d49',
 'b248a173c88030e2',
 'e81cef823d61598e',
 'eb51d995053227fe',
 '938b366e21867b02',
 '54316f9894f581a5',
 'f71b7e9ee08f752d',
 '27a974d4252be2c5',
 'cd9643c2361c4be1',
 '47da69eed139fd91',
 '907b705ebf37dd20',
 '0721dca7ddbb052a',
 '396a7de23f4678d1',
 '4c1cbc2f27bebb23',
 '5c98b8b4e4736fb8',
 '1034f971ba302cfe',
 '1d46797a91f23226',
 '228bfe4c3193b637',
 'cd0f98c3ac5dc90d',
 'f63e54efdebb676d',
 '8e628a76c41dcd03',
 'cb413b4dbbbcf64d',
 'e3a8976e3b65d401',
 '8284ccd643901a63',
 'a58aac324cb80dfc',
 '10c41e9532dfbdff',
 '1c922bc18c4005ec',
 '20736842b0f8b4af',
 '7dbc16debe8b073b',
 '510b747b05412e11',
 'c5f646ff27a8941c',
 'a2560185f1b5a502',
 'bf134c7210c5f8ef',
 'ce60504b00634f19',
 '35c07a2b44fab297',
 'c159614190de58b4',
 'd08de9949ed2c401',
 '47b23fb868b86f50',
 'c4e80b6e80feaac4',
 'a778675809db53a0',
 '84a78e4a67d52901',
 '2d99e55adcf98c62',
 'e2cf6f9b14fa0a06',
 '1b61675364000090',
 'a368adb2d753b60d',
 '9b0dc0a845e8111f',
 'b940f9fdf78d42bf',
 '8cca5b76b002ab1f',
 'fd460a6998eb0990',
 'c8ce1cb46d04e01b',
 '2a6a715cb2a8e2d9',
 'ce84945e21483152',
 'ce51e400b1343abb',
 '25c44d94e5e00812',
 '2d54f897673aed88',
 '2c49aeb37a2bc372',
 '9078bff10854a8da',
 '6b56c66e1b71780d',
 '5982d2877fc8af57',
 'f30a03b2f3c70aae',
 '5b8d38dcc661955b',
 'de16d37b1a5cefb8',
 '350ef35527b6d769',
 '2662296cf2f64a8c',
 '71575b369c8ce89e',
 'd7f23a59f4a2cbef',
 '949a6810e73170eb',
 '82c58a28acec2695',
 'c2479d4d929321fb',
 '8ab88a6a10f6ff37',
 '9c8d19417d5a63e5',
 'aab7dd131da4a257',
 '8e6f669e45a2c534',
 '23feb6c1fbb5d068',
 '63aa3c7a93c048fe',
 'fe3f9c0c49a5b05b',
 'edc93d0bb89bb1c6',
 '01e27ad07494a6c4',
 '9da3eb70e91dd073',
 '631620dd981ec797',
 '2c1697ec2197043e',
 '0f246f743208056d',
 '0689dec28386f250',
 '46a08676a1f97e22',
 '72aec8198f1239e1',
 'bf57f8d8e3a5ffac',
 '77fc22f0005af9ad',
 '7f5f4d8620441517',
 '9c75381fd8e05940',
 'c43e000083d8087f',
 '54d3c19829ae9e4b',
 '561c23e2e3551e3b',
 'e11916314bd0d35e',
 'b591b8e64bde8c97',
 '7ded26a66c16e008',
 '5d80231268e847be',
 'da041ce896f51cef',
 '4fbf9c5301f99cf4',
 '039b881fafb281fc',
 'fa935b600f0ed202',
 'f84f4d8ba6e05d94',
 '6505d148f095f945',
 'cad7f3a0a8516150',
 'c435fc7260cd1d25',
 '964b231f16164b48',
 '2c4d0ec25d5e5772',
 '6cb6d5ffcb406792',
 '2bbf1f990caa78d6',
 'a19ae8444e0f33e7',
 '4af5d05941e99f30',
 '1b406ba8be2030ac',
 '421e9e87bddb9aaa',
 'e096b8d9a27132d5',
 '952734942f3d0425',
 'f4d744dbd88cba81',
 '272c12eea6a66d00',
 '337a877c9f956fdf',
 'ce8ee606a1d20641',
 '645c32f2de0d9ff1',
 '2924fbb3026dce6e',
 '7e0982c1c55eaf94',
 'a299978c84cccbb0',
 '2170f27baa83b806',
 '7b3a75e7ddc9526d',
 'e67aafb6096abe42',
 '024c4c5fcbd32a79',
 '9ed92a95b76a4035',
 '1ef2d1b196119e42',
 '653f8c23147f535c',
 'a6de013b26b7b117',
 'dfb94d5621a7ae15',
 'd1668fcf4040b81a']
df = pd.read_csv('../input/train.csv')
df_problems = df[df['id'].isin(train_image_ids)]
print('total: ', len(df_problems))
import matplotlib.pyplot as plt
from IPython.display import Image
from IPython.core.display import HTML 
Image(url=df_problems.iloc[0].url, width=100, height=100)


for id_, url, landmark_id in df_problems.sample(100).values:
    print(id_, landmark_id, url)



test_df = pd.read_csv('../input/test.csv')
test_df_problems = test_df[test_df['id'].isin(test_image_ids)]
for id_, url in test_df_problems.values:
    print(id_, url)


