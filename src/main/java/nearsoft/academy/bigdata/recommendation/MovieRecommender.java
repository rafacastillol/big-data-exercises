package nearsoft.academy.bigdata.recommendation;


import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.impl.neighborhood.ThresholdUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.UserBasedRecommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.apache.mahout.common.iterator.FileLineIterator;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;

public class MovieRecommender {
  private long totalReviews;
  private Map<String, Long> users;
  private Map<Long, String> products;
  private UserBasedRecommender recommender;
  

  public MovieRecommender(String path) throws IOException, TasteException {
    this.totalReviews = 0;
    this.users = new HashMap<String, Long>();

    DataModel model = this.loadReviews(path);
    UserSimilarity similarity = new PearsonCorrelationSimilarity(model);
    UserNeighborhood neighborhood = new ThreadedThresholdUserNeighborhood(0.1, similarity, model, 5);
    this.recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);
  }

  private DataModel loadReviews(String path) throws IOException {
    File file = new File(path);
    FileLineIterator iterator = new FileLineIterator(file);
    BiMap<String, Long> products = HashBiMap.create();

    long nextUserID = 0;
    long nextProductID = 0;

    long userID = 0;
    long productID = 0;
    float score = 0.0f;
    // We can turn this into a DataModel later
    FastByIDMap<Collection<Preference>> data = new FastByIDMap<Collection<Preference>>();
    while (iterator.hasNext()) {
      String line = iterator.next();

      if (line.length() < 9) {
        continue;
      }
      String prefix = line.substring(0, 9);
      switch (prefix) {
        case "product/p":
          String productString = line.substring(19);

          if (!products.containsKey(productString)) {
            products.put(productString, nextProductID++);
          }
          productID = products.get(productString);
          break;
        case "review/us":
          String userString = line.substring(15);

          if (!this.users.containsKey(userString)) {
            users.put(userString, nextUserID++);
          }
          userID = users.get(userString);

          break;
        case "review/sc":
          score = Float.parseFloat(line.substring(14));
          Collection<Preference> prefs = data.get(userID);
          if (prefs == null) {
            prefs = new ArrayList<Preference>(16);
            data.put(userID, prefs);
          }
          Preference pref = new GenericPreference(userID, productID, score);
          prefs.add(pref);
          totalReviews++;
          break;
      }
    }

    // we don't really need to look up numerical ids anymore.
    this.products = products.inverse();
    
    return new GenericDataModel(GenericDataModel.toDataMap(data, true));
  }

  public long getTotalReviews() {
    return this.totalReviews;
  }

  public long getTotalProducts() {
    return this.products.size();
  }

  public long getTotalUsers() {
    return this.users.size();
  }

  public List<String> getRecommendationsForUser(String userID) throws TasteException {
    long numericUserID = this.users.get(userID);

    List<RecommendedItem> recommendations = this.recommender.recommend(numericUserID, 3);
    List<String> recommendationIDs = new ArrayList<String>(recommendations.size());

    for (RecommendedItem recommendation: recommendations) {
      recommendationIDs.add(this.products.get(recommendation.getItemID()));
    }

    return recommendationIDs;
  }
}
