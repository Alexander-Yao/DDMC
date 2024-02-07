import time
import warnings
warnings.filterwarnings("ignore")
from utils.multi_aug_data_loader import MyDataset
from sklearn.cluster import KMeans
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
import numpy as np
import torch
from torchvision import datasets, transforms
import random
from torch.utils.data import Dataset, DataLoader
from torch import optim
from multi_vae.DMCModels import VAE
from multi_vae.DMCTraining import Trainer
from sklearn.metrics import accuracy_score, normalized_mutual_info_score as nmi, rand_score as ri, adjusted_rand_score as ar
from parse import get_parse

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(1)

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
transform_list = [
    transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ]),
    # transforms.Compose([
    #     transforms.Resize((64, 64)),
    #     # transforms.RandomResizedCrop(224),
    #     transforms.RandomRotation(10),
    #     transforms.ToTensor(),
    # ]),
    transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ColorJitter(hue=0.5),
        transforms.ToTensor(),
    ])
]

args = get_parse()
# 创建一个datasets.ImageFolder的对象，传入数据集的路径
num_clusterings = args.K
num_clusters = args.M
dataset = datasets.ImageFolder(f"datasets/{args.dataset}/instance")
batch_size = args.batch_size
cluster_dict = {"fruit": ["color", "species"],
                "fruit360": ["color", "species"],
                "cards": ["number", "suits"]}

# 创建一个自定义的数据集对象，传入datasets.ImageFolder的对象和transform列表
my_dataset = MyDataset(dataset, transforms=transform_list)

my_dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=False)

NMI_c = []
NMI_cz = []
ACC_c = []
ACC_cz = []

dataset_name = args.dataset
settings = [[1, 32], [1, 64]]  # share autoencoder, Batch_size
iters_to_add_capacity = [25000, 25000, 25000, 25000, 25000, 25000]
for d in [0]:  # datasets index
    DATA = dataset_name
    share = settings[d][0]
    Batch_size = settings[d][1]
    iters_add_capacity = iters_to_add_capacity[d]
    Epochs = args.epoch
    lr = 5e-4
    Net = 'C'  # CNN
    hidden_dim = 256
    z_variables = 10
    start_time = time.time()
    # for beta in [10, 20, 30, 40, 50]:
    #     for capacity in [3, 4, 5, 6, 7]:
    runs = 1
    TEST = False
    for beta in [30]:
        for capacity in [5]:
            ACCc = 0
            NMIc = 0
            ARIc = 0
            PURc = 0
            ACCcz = 0
            NMIcz = 0
            ARIcz = 0
            PURcz = 0
            use_cuda = torch.cuda.is_available()
            # use_cuda = False
            print("cuda is available?", use_cuda)
            for i in range(runs):
                model_name = DATA + '.pt'
                print('Run:' + str(i))
                if Net == 'C':
                    train_loader, n_dis, n_clusterings, size = my_dataloader, len(transform_list), num_clusterings, my_dataset.__len__()
                    print(train_loader, n_dis, n_clusterings, size)
                    print('Iters:' + str(size / Batch_size * Epochs))
                    cont_capacity = [capacity, beta, iters_add_capacity]
                    disc_capacity = [np.log(n_clusterings), beta, iters_add_capacity]

                latent_spec = {'cont': z_variables,
                               'disc': [n_clusterings]}

                img_size=(3, 64, 64)
                # img_size=(3, 32, 32)
                # img_size = (1, 32, 32)
                if TEST == False:
                    # Build a model

                    model = VAE(latent_spec=latent_spec, img_size=img_size,
                                view_num=len(transform_list), use_cuda=use_cuda,
                                Network=Net, hidden_dim=hidden_dim, shareAE=share, num_clusters=num_clusters)
                    if use_cuda:
                        model.cuda()

                    print(model)

                    # Train the model

                    # Build optimizer
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    optimizer2 = optim.Adam([model.v[0].weight], lr=lr)


                    # Build a trainer
                    trainer = Trainer(model, optimizer, optimizer2,
                                      cont_capacity=cont_capacity, disc_capacity=disc_capacity, view_num=n_dis,
                                      use_cuda=use_cuda, DATA=DATA, num_clusters=num_clusters)

                    # Train model for Epochs
                    trainer.train(train_loader, epochs=Epochs)
                    # torch.save(trainer.model.state_dict(), './models/' + model_name)
                    # print('save model?')
                    TEST = True

                if TEST == True:
                    # path_to_model_folder = './models/' + model_name
                    batch_size_test = 140000  # max number to cover all dataset.
                    # if Net == 'C':
                    #     train_loader, n_dis, n_clusters, _ = Get_dataloaders(batch_size=batch_size_test,
                    #                                                             DATANAME=DATA + '.mat')
                    # model = load(latent_spec=latent_spec,
                    #              path=path_to_model_folder,
                    #              n_dis=n_dis,
                    #              img_size=img_size,
                    #              Network=Net,
                    #              hid=hidden_dim, shareAE=share)

                    # Print the latent distribution info
                    print(model.MvLatent_spec)

                    # Print model architecture
                    # print(model)

                    # for i in range(n_dis):
                    #     img_save = datasets.ImageFolder(root='datasets/fruit_type/color', transform=transforms[i])
                    #     imgLoader_save = torch.utils.data.DataLoader(img_save, batch_size=1, shuffle=False, num_workers=1)
                    #
                    #     label_list = []
                    #     embedding_list = []
                    #     for batch_index, (images, labels) in enumerate(imgLoader_save):
                    #         outputs = cnn.get_embedding(images.to(device))
                    #         embedding_list.extend(outputs.cpu().detach().numpy())
                    #         label_list.extend(labels.cpu().numpy())
                    #     saved_embeddings = np.array(embedding_list)
                    #     saved_labels = np.array(label_list)
                    #     if args.save:
                    #         print('save color', saved_embeddings.shape, saved_labels.shape)
                    #         np.save(os.path.join(embedding_path, 'embedding_color.npy'), saved_embeddings)
                    #         np.save(os.path.join(embedding_path, 'label_color.npy'), saved_labels)
                    #
                    #     color_nmi_score = []
                    #     color_ar_score = []
                    #     color_ri_score = []
                    #     for i in range(clustering_times):
                    #         kmeans = KMeans(n_clusters=3, random_state=i).fit(saved_embeddings)
                    #         pred_res = kmeans.labels_
                    #         color_nmi_score.append(nmi(saved_labels, pred_res))
                    #         color_ar_score.append(ar(saved_labels, pred_res))
                    #         color_ri_score.append(ri(saved_labels, pred_res))
                    #     color_nmi_score = max(color_nmi_score)
                    #     color_ar_score = max(color_ar_score)
                    #     color_ri_score = max(color_ri_score)

                    # clustering results
                    # label_list = [[] for _ in range(num_clusterings)]
                    print("name", dataset_name)
                    clustering_path = cluster_dict[dataset_name]
                    each_cluster_num = [3, 3]
                    for _i in range(num_clusterings):
                        embedding_list = [[] for _ in range(n_dis)]
                        this_dataset = datasets.ImageFolder(f"datasets/{dataset_name}/type/" + clustering_path[_i])
                        label_list = []

                        # 创建一个自定义的数据集对象，传入datasets.ImageFolder的对象和transform列表
                        this_my_dataset = MyDataset(this_dataset, transforms=transform_list)

                        this_dataloader = DataLoader(this_my_dataset, batch_size=batch_size, shuffle=False)
                        for batch_idx, Data in enumerate(this_dataloader):
                            # encode data
                            labels = Data[-1]
                            # print('label', labels)
                            inputs = []
                            this_data = Data[0].cuda() if use_cuda else Data[0]
                            data = torch.split(this_data, 1, dim=1)
                            data = [torch.squeeze(_, dim=1) for _ in data]
                            for i in range(n_dis):
                                inputs.append(data[i])
                            encodings = model.encode(inputs)
                            # clustering common embedding
                            x = encodings['disc'][0].cpu().detach().data.numpy()
                            multiview_z = []
                            multiview_cz = []
                            for i in range(n_dis):
                                name = 'cont' + str(i + 1)
                                # Continuous encodings, view-peculiar variables
                                x_c = encodings[name][0].cpu().detach().data.numpy()  # z, single disentangled feature
                                xi = min_max_scaler.fit_transform(x_c)  # scale to [0,1]
                                multiview_z.append(np.concatenate([xi, x], axis=1))  # z + c
                                multiview_cz.append(xi)
                                # print(multiview_z[-1].shape)
                                embedding_list[i].extend(xi)
                                # print(multiview_z[-1][0])
                            y = labels.cpu().detach().data.numpy()
                            label_list.extend(y)
                        # print(len(label_list), label_list)

                        for i in range(n_dis):
                            kmeans = KMeans(n_clusters=each_cluster_num[_i], n_init=100).fit(embedding_list[i])
                            pred_label = kmeans.labels_
                            print('nmi for clustering', clustering_path[_i], '{}-th embedding'.format(i), nmi(label_list, pred_label))


                        # Discrete encodings, view-common variable
                        # x = encodings['disc'][0].cpu().detach().data.numpy()
                        # multiview_z = []
                        # multiview_cz = []
                        # for i in range(n_dis):
                        #     name = 'cont' + str(i + 1)
                        #     # Continuous encodings, view-peculiar variables
                        #     x_c = encodings[name][0].cpu().detach().data.numpy()  # z
                        #     xi = min_max_scaler.fit_transform(x_c)  # scale to [0,1]
                        #     multiview_z.append(np.concatenate([xi, x], axis=1))  # z + c
                        #     multiview_cz.append(xi)
                        #     print(multiview_z[-1].shape)
                        #     print(multiview_z[-1][0])
                        # y = labels.cpu().detach().data.numpy()

                        # p = kmeans.fit_predict(x)
                        # print('k-means on C')
                        # print(x.shape)
                        #
                        # test(y, p)
                        # p = x.argmax(1)
                        # print('Multi-VAE-C: y = C.argmax(1)')
                        # test(y, p)
                        # ACCc += Nmetrics.acc(y, p)
                        # NMIc += Nmetrics.nmi(y, p)
                        # ARIc += Nmetrics.ari(y, p)
                        # PURc += Nmetrics.purity(y, p)
                        #
                        # X_all = np.concatenate(multiview_cz, axis=1)
                        # p = kmeans.fit_predict(X_all)
                        # print('k-means on [z1, z2, ..., zV]')
                        # print(X_all.shape)
                        # test(y, p)
                        # print('k-means on [zv]\nk-means on [C, zv]')
                        # print(multiview_cz[0].shape, multiview_z[0].shape)
                        # for i in range(n_dis):
                        #     name = 'cont' + str(i + 1)
                        #     x_cz = encodings[name][0].cpu().detach().data.numpy()
                        #     x_Conz = multiview_z[i]
                        #     p = kmeans.fit_predict(x_cz)
                        #     test(y, p)
                        #     p = kmeans.fit_predict(x_Conz)
                        #     test(y, p)
                        #     print('\n')
                        #
                        # multiview_cz.append(x)
                        # X_all = np.concatenate(multiview_cz, axis=1)
                        # p = kmeans.fit_predict(X_all)
                        # # scio.savemat('./viz/' + str(Epochs) + '.mat', {'Z': X_all, 'Y': y, 'P': p})
                        # print('Multi-VAE-CZ: k-means on [C, z1, z2, ..., zV]')
                        # print(X_all.shape)
                        # test(y, p)
                        # ACCcz += Nmetrics.acc(y, p)
                        # NMIcz += Nmetrics.nmi(y, p)
                        # ARIcz += Nmetrics.ari(y, p)
                        # PURcz += Nmetrics.purity(y, p)
                        #
                        # print(_i, 'Multi-VAE-C:', ACCc / runs, NMIc / runs, ARIc / runs, PURc / runs)
                        # print(_i, 'Multi-VAE-CZ:', ACCcz / runs, NMIcz / runs, ARIcz / runs, PURcz / runs)
            # np.save('Cmetics.npy', [ACCc/runs, NMIc/runs, ARIc/runs, PURc/runs])
            # np.save('CZmetics.npy', [ACCcz/runs, NMIcz/runs, ARIcz/runs, PURcz/runs])
            # NMI_c.append(NMIc / runs)
            # NMI_cz.append(NMIcz / runs)
            # ACC_c.append(ACCc / runs)
            # ACC_cz.append(ACCcz / runs)
    # end_time = time.time()
    # print('Total time', start_time - end_time)
    # print(NMI_c)
    # np.save('NMI_c.npy', NMI_c)
    # print(NMI_cz)
    # np.save('NMI_cz.npy', NMI_cz)
    # print(ACC_c)
    # np.save('ACC_c.npy', ACC_c)
    # print(ACC_cz)
    # np.save('ACC_cz.npy', ACC_cz)
